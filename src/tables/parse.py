from typing import List, Literal, Union, TypedDict
from pydantic import BaseModel, Field

from src.schemas import TableElement, Bbox
from src.pdf import Pdf


class ParsingArgs(BaseModel):
    parsing_algorithm: str
    table_output_format: Literal["str", "markdown", "html"] = Field(default="str")


class TableTransformersArgs(ParsingArgs):
    min_table_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    min_cell_confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    parsing_algorithm: Literal["table-transformers"] = Field(
        default="table-transformers"
    )


class PyMuPDFArgs(ParsingArgs):
    parsing_algorithm: Literal["pymupdf"] = Field(default="pymupdf")


def _ingest_with_pymupdf(
    doc: Pdf,
    parsing_args: PyMuPDFArgs,
) -> List[TableElement]:
    raise NotImplementedError("PyMuPDF table parsing is not yet implemented.")


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

            tables.append(
                TableElement(
                    bbox=Bbox(
                        page=page_num,
                        x0=table_bbox.bbox[0],
                        y0=table_bbox.bbox[1],
                        x1=table_bbox.bbox[2],
                        y1=table_bbox.bbox[3],
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
