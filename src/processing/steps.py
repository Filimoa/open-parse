import statistics
from collections import defaultdict
from typing import Literal, Type, List, Union, Sequence
from pathlib import Path

import asyncio
import fitz  # type: ignore
from loguru import logger

from src.app import consts
from ..exceptions import ParsingFailureException
from . import tables
from .pages import get_metadata_for_each_page
from .schemas import (
    TableElement,
    TextElement,
    LineElement,
    FileMetadata,
    NodeVariant,
    PageMetadata,
    ParsedDoc,
)

# due to openai rate limits
MAX_PAGES_FOR_TABLES = 200


def _split_large_elements(elements: list[TextElement]) -> list[TextElement]:
    PIXELS_IN_TAB = 2
    STUB_LENGTH_PCT = 0.3
    res = []
    for element in elements:
        if not element.is_large:
            res.append(element)
            continue

        median_x0 = statistics.median([l.x0 for l in element.lines])
        median_len = statistics.median([len(l.text) for l in element.lines])
        split_points = []

        for line in element.lines:
            if line.x0 > median_x0 + PIXELS_IN_TAB:
                split_points.append(line.y0)
            elif len(line.text) < median_len * STUB_LENGTH_PCT:
                split_points.append(line.y1)

        if len(split_points) == 0:
            res.append(element)
            continue

        res.extend(element.split(split_points))
    return res


def _group_elements_by_page(
    elements: list[TextElement],
) -> dict[int, list[TextElement]]:
    grouped_elements = defaultdict(list)
    for element in elements:
        grouped_elements[element.page].append(element)

    for k, v in grouped_elements.items():
        grouped_elements[k] = sorted(v, key=lambda x: x.position, reverse=True)

    return grouped_elements


def _combine_elements_spatially(
    elements: list[TextElement],
    x_error_margin: float = 0,
    y_error_margin: float = 0,
    critera: Literal["both_small", "either_stub"] = "both_small",
) -> list[TextElement]:
    # Look into implementing IntervalTree
    grouped_elements = _group_elements_by_page(elements)

    combined_all = []
    for page_elements in grouped_elements.values():
        combined: list[TextElement] = []
        for element in page_elements:
            for other in combined:
                if critera == "both_small":
                    critera_bool = element.is_small and other.is_small
                elif critera == "either_stub":
                    critera_bool = element.is_stub or other.is_stub
                if (
                    element.overlaps(
                        other,
                        x_error_margin=x_error_margin,
                        y_error_margin=y_error_margin,
                    )
                    and critera_bool
                ):
                    combined.remove(other)
                    combined.append(element.combine(other))
                    break
            else:
                combined.append(element)

        combined_all.extend(combined)

    return combined_all


def _combine_bullets(elements: list[TextElement]) -> list[TextElement]:
    grouped_elements = _group_elements_by_page(elements)

    combined_all = []
    for page_elements in grouped_elements.values():
        combined: list[TextElement] = []
        for element in page_elements:
            for other in combined:
                if (
                    element.neighbors(other)
                    and element.is_bullet
                    and other.is_bullet
                    and element.tokens + other.tokens < consts.TOKENIZATION_UPPER_LIMIT
                ):
                    combined.remove(other)
                    combined.append(element.combine(other))
                    break

            else:
                combined.append(element)

        combined_all.extend(combined)

    # maybe sort everything by position here?
    return combined_all


def _remove_metadata_elements(
    elements: list[TextElement], page_height: int
) -> list[TextElement]:
    # looking to remove page numbers, headers, footers, etc.
    res = []
    for e in elements:
        if e.y0 > page_height * 0.12 and e.y1 < page_height * 0.88:
            res.append(e)
        elif not e.is_stub:
            res.append(e)
    return res


def _filter_out_repeated_elements(
    elements: list[TextElement], threshold: int = 2
) -> list[TextElement]:
    text_counts: dict[str, int] = defaultdict(int)
    for element in elements:
        if element.text is not None:
            text_counts[element.text] += 1

    repeated_texts = set(
        text for text, count in text_counts.items() if count > threshold
    )

    filtered_elements = []
    for element in elements:
        if element.text is None or element.text not in repeated_texts:
            filtered_elements.append(element)

    return filtered_elements


def _filter_out_full_page_elements(
    elements: list[TextElement], document_area: float
) -> list[TextElement]:
    res = []
    for e in elements:
        if e.area / document_area < 0.5:
            res.append(e)
        elif not e.is_stub:
            res.append(e)

    return res


def _filter_out_short_elements(elements: list[TextElement]) -> list[TextElement]:
    return [e for e in elements if not e.is_stub]


def create_line_elements(lines: dict, error_margin: float = 5) -> list[LineElement]:
    """Creates LineElement objects from given lines, combining overlapping ones."""
    combined: list[LineElement] = []

    for line in lines:
        bbox = line["bbox"]
        text = " ".join(span["text"] for span in line["spans"])

        line_element = LineElement(
            x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3], text=text
        )
        for i, other in enumerate(combined):
            if line_element.overlaps(
                other, error_margin=error_margin
            ) and line_element.same_level(other, error_margin=error_margin):
                combined[i] = line_element.combine(other)
                break
        else:
            combined.append(line_element)

    return combined


def _out_of_bounds(
    x0: float, y0: float, x1: float, y1: float, pdf_width: float, pdf_height: float
) -> bool:
    """Checks if a node is out of bounds of the pdf document."""
    return x0 > pdf_width or x1 > pdf_width or y0 > pdf_height or y1 > pdf_height


def _parse_elements(pdf: fitz.Document) -> list[TextElement]:
    """Parses text elements from a given pdf document."""
    elements = []
    for page in pdf:
        page_ocr = page.get_textpage_ocr(flags=0, full=False)
        for node in page.get_text("dict", textpage=page_ocr, sort=True)["blocks"]:
            if node["type"] != 0:
                continue
            if _out_of_bounds(
                x0=node["bbox"][0],
                y0=node["bbox"][1],
                x1=node["bbox"][2],
                y1=node["bbox"][3],
                pdf_width=pdf[0].rect.width,
                pdf_height=pdf[0].rect.height,
            ):
                continue
            lines = create_line_elements(node["lines"])
            elements.append(
                TextElement(
                    x0=node["bbox"][0],
                    y0=node["bbox"][1],
                    x1=node["bbox"][2],
                    y1=node["bbox"][3],
                    page=page.number,
                    position=node["number"],
                    text="\n".join(line.text for line in lines),
                    lines=lines,
                    variant=NodeVariant.TEXT,
                )
            )
    return elements


def _combine_text_and_table_elements(
    text_elements: list[TextElement], table_elements: list[TableElement]
) -> Sequence[TextElement | TableElement]:
    """Combines text and table elements into a single list. Remove any text elements that are found on a page with a table (for now)"""
    filtered_text_elems: list[TextElement] = []
    pages_with_tables = {table.page for table in table_elements}
    for element in text_elements:
        if element.page not in pages_with_tables:
            filtered_text_elems.append(element)
    return filtered_text_elems + table_elements  # type: ignore


def _get_page_metadata(pdf: fitz.Document) -> Sequence[PageMetadata]:
    document_area = pdf[0].rect.width * pdf[0].rect.height

    res = _parse_elements(pdf)
    res = _filter_out_full_page_elements(res, document_area)
    return get_metadata_for_each_page(res, pdf)


def _transform_elements(
    pdf: fitz.Document, summarize_tables: bool = False
) -> Sequence[TextElement | TableElement]:
    document_area = pdf[0].rect.width * pdf[0].rect.height

    res = _parse_elements(pdf)
    res = _filter_out_full_page_elements(res, document_area)

    # intial merging
    res = _combine_elements_spatially(res, x_error_margin=4, y_error_margin=4)
    res = _combine_elements_spatially(res)
    res = _combine_bullets(res)

    # dealing with stubs
    res = _remove_metadata_elements(res, pdf[0].rect.height)
    res = _combine_elements_spatially(
        res, x_error_margin=4, y_error_margin=12, critera="either_stub"
    )
    res = _combine_elements_spatially(res, critera="either_stub")
    res = _filter_out_short_elements(res)

    # misc cleanup
    res = _split_large_elements(res)
    res = _filter_out_repeated_elements(res)
    res = _standardize_positions(res)

    # parse tables
    res_with_tables: Sequence[TextElement | TableElement] = res
    parsed_tables = []
    if pdf.page_count > MAX_PAGES_FOR_TABLES and summarize_tables:
        logger.info(f"Skipping table parsing due to page count {pdf.page_count}")
    else:
        parsed_tables = asyncio.run(
            tables.parse_tables(pdf, generate_summary=summarize_tables)
        )

    res_with_tables = _combine_text_and_table_elements(res, parsed_tables)

    res_with_tables = _standardize_positions(res_with_tables)

    return res_with_tables


def _standardize_positions(
    elements: Sequence[TextElement | TableElement],
) -> Sequence[TextElement | TableElement]:
    sorted_elements = sorted(elements, key=lambda x: (x.page, x.position))

    for ix, element in enumerate(sorted_elements):
        element.position = ix

    return sorted_elements


def parse_elements(pdf_bytes: bytes) -> Sequence[TextElement | TableElement]:
    try:
        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
        if pdf.is_encrypted:
            raise ParsingFailureException("PDF is encrypted")
        elements = _transform_elements(pdf)
    except ValueError:
        raise ParsingFailureException("Error while parsing pdf")

    return elements


def parse(file: str | Path, ai_table_summaries: bool = False) -> ParsedDoc:
    pdf = fitz.open(file)

    elements = _transform_elements(pdf, summarize_tables=ai_table_summaries)
    metadata = _get_page_metadata(pdf)

    return ParsedDoc(
        elements=elements,
        page_metadata=metadata,
        file_metadata=FileMetadata(
            num_pages=pdf.page_count,
            filename=pdf.name,
        ),
    )
