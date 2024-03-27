from typing import List, Literal, Union, TypedDict
from pydantic import BaseModel, Field, ConfigDict

from src.schemas import TableElement, Bbox
from src.pdf import Pdf
from . import pymupdf


class ParsingArgs(BaseModel):
    parsing_algorithm: str
    table_output_format: Literal["str", "markdown", "html"] = Field(default="html")


class TableTransformersArgs(ParsingArgs):
    min_table_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_cell_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    parsing_algorithm: Literal["table-transformers"] = Field(
        default="table-transformers"
    )

    model_config = ConfigDict(extra="forbid")


class PyMuPDFArgs(ParsingArgs):
    parsing_algorithm: Literal["pymupdf"] = Field(default="pymupdf")

    model_config = ConfigDict(extra="forbid")


def _ingest_with_pymupdf(
    doc: Pdf,
    parsing_args: PyMuPDFArgs,
) -> List[TableElement]:
    pdoc = doc.to_pymupdf_doc()
    tables = []
    for page_num, page in enumerate(pdoc):
        tabs = page.find_tables()
        for i, tab in enumerate(tabs.tables):
            headers = tab.header.names
            lines = tab.extract()

            if parsing_args.table_output_format == "str":
                text = pymupdf.output_to_markdown(headers, lines)
            elif parsing_args.table_output_format == "markdown":
                text = pymupdf.output_to_markdown(headers, lines)
            elif parsing_args.table_output_format == "html":
                text = pymupdf.output_to_html(headers, lines)

            # Flip y-coordinates to match the top-left origin system
            bbox = pymupdf.combine_header_and_table_bboxes(tab.bbox, tab.header.bbox)
            fy0 = page.rect.height - bbox[3]
            fy1 = page.rect.height - bbox[1]

            table = TableElement(
                bbox=Bbox(
                    page=page_num,
                    x0=bbox[0],
                    y0=fy0,
                    x1=bbox[2],
                    y1=fy1,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                ),
                text=text,
            )
            tables.append(table)
    return tables


def _ingest_with_table_transformers(
    doc: Pdf,
    args: TableTransformersArgs,
) -> List[TableElement]:
    try:
        from src.tables.utils import doc_to_imgs
        from .table_transformers.ml import find_table_bboxes, get_table_content
    except ImportError as e:
        raise ImportError(
            "Table detection and extraction requires the `torch`, `torchvision` and `transformers` libraries to be installed."
        )
    pdoc = doc.to_pymupdf_doc()  # type: ignore
    pdf_as_imgs = doc_to_imgs(pdoc)

    pages_with_tables = {}
    for page_num, img in enumerate(pdf_as_imgs):
        pages_with_tables[page_num] = find_table_bboxes(img, args.min_table_confidence)

    tables = []
    for page_num, table_bboxes in pages_with_tables.items():
        page = pdoc[page_num]
        page_dims = (page.rect.width, page.rect.height)
        for table_bbox in table_bboxes:
            table = get_table_content(
                page_dims, img, table_bbox.bbox, args.min_cell_confidence
            )
            table._run_ocr(page)

            if args.table_output_format == "str":
                table_text = table.to_str()
            elif args.table_output_format == "markdown":
                table_text = table.to_markdown_str()
            elif args.table_output_format == "html":
                table_text = table.to_html_str()

            # Flip y-coordinates to match the top-left origin system
            fy0 = page.rect.height - table_bbox.bbox[3]
            fy1 = page.rect.height - table_bbox.bbox[1]

            tables.append(
                TableElement(
                    bbox=Bbox(
                        page=page_num,
                        x0=table_bbox.bbox[0],
                        y0=fy0,
                        x1=table_bbox.bbox[2],
                        y1=fy1,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    ),
                    text=table_text,
                )
            )

    return tables


def ingest(
    doc: Pdf,
    parsing_args: Union[TableTransformersArgs, PyMuPDFArgs, None] = None,
) -> List[TableElement]:
    if isinstance(parsing_args, TableTransformersArgs):
        return _ingest_with_table_transformers(doc, parsing_args)
    elif isinstance(parsing_args, PyMuPDFArgs):
        return _ingest_with_pymupdf(doc, parsing_args)
    else:
        raise ValueError(f"Unsupported parsing_algorithm.")
