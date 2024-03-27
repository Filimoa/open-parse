from typing import List, Literal, Union, TypedDict
from pydantic import BaseModel, Field

from src.schemas import TableElement, Bbox
from src.pdf import Pdf


class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["str", "markdown", "html"]


class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["str", "markdown", "html"]


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


def args_dict_to_model(
    args_dict: Union[
        TableTransformersArgsDict,
        PyMuPDFArgsDict,
        None,
    ]
) -> ParsingArgs:
    if args_dict is None:
        args_dict = PyMuPDFArgsDict()
    parsing_algorithm = args_dict.get("parsing_algorithm", "table-transformers")

    if parsing_algorithm == "table-transformers":
        return TableTransformersArgs(**args_dict)
    elif parsing_algorithm == "pymupdf":
        return PyMuPDFArgs(**args_dict)
    else:
        raise ValueError(f"Unsupported parsing_algorithm: {parsing_algorithm}")


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
    parsing_args: Union[TableTransformersArgsDict, PyMuPDFArgsDict, None] = None,
) -> List[TableElement]:
    args = args_dict_to_model(parsing_args)
    if isinstance(args, TableTransformersArgs):
        return _ingest_with_table_transformers(doc, args)
    elif isinstance(args, PyMuPDFArgs):
        return _ingest_with_pymupdf(doc, args)
    else:
        raise ValueError(f"Unsupported parsing_algorithm: {args.parsing_algorithm}")
