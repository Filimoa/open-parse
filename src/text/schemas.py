from typing import Optional
import re
import fitz

from src.schemas import TableElement, PageMetadata, TextElement


def find_page_number(text: str) -> Optional[int]:
    re_match = re.search(r"page\s+(\d+)", text, re.IGNORECASE)
    if re_match:
        return int(re_match.group(1))
    else:
        return None


def len_no_extra_spaces(text: str) -> int:
    """Return the length of the string with extra spaces removed."""
    return len(re.sub(r"\s{2,}", " ", text))


def _get_single_page_metadata(
    elems: list[TextElement], pdf_page: fitz.Page
) -> PageMetadata:
    unique_pages = set([e.page for e in elems])
    assert (
        len(unique_pages) == 1
    ), f"All elements must be from the same page. Found {unique_pages}"

    page_label, metadata = None, None
    for e in elems:
        if "page" in e.text.lower() and len_no_extra_spaces(e.text) < 50:
            page_label = find_page_number(e.text.lower())
            break

    for e in elems:
        if "attachment" in e.text.lower() and len_no_extra_spaces(e.text) < 50:
            metadata = e.text.lower().strip().replace("\n", ", ")
            break

    return PageMetadata(
        page_label=page_label,
        metadata=metadata,
        page_height=pdf_page.rect.width,
        page_width=pdf_page.rect.height,
        page_num=pdf_page.number,
    )


def _group_elements_by_page(
    elements: list[TextElement],
) -> dict[int, list[TextElement]]:
    page_elements: dict[int, list[TextElement]] = {}
    for e in elements:
        if e.page not in page_elements:
            page_elements[e.page] = []
        page_elements[e.page].append(e)

    return page_elements


def get_metadata_for_each_page(
    elements: list[TextElement], pdf: fitz.Document
) -> list[PageMetadata]:
    page_elements = _group_elements_by_page(elements)
    page_metadata = []
    for page, elems in page_elements.items():
        metadata = _get_single_page_metadata(elems, pdf_page=pdf.load_page(page))
        page_metadata.append(metadata)

    return page_metadata
