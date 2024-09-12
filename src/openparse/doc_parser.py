from pathlib import Path
from typing import List, Literal, TypedDict, TypeVar, Union

from openparse import consts, tables, text
from openparse._types import NOT_GIVEN, NotGiven
from openparse.pdf import Pdf
from openparse.processing import (
    BasicIngestionPipeline,
    IngestionPipeline,
    NoOpIngestionPipeline,
)
from openparse.schemas import Node, ParsedDocument, TableElement, TextElement

from openparse.schemas import ImageElement

IngestionPipelineType = TypeVar("IngestionPipelineType", bound=IngestionPipeline)


class UnitableArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["unitable"]
    min_table_confidence: float
    table_output_format: Literal["html"]


class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["markdown", "html"]


class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["markdown", "html"]


def _table_args_dict_to_model(
    args_dict: Union[TableTransformersArgsDict, PyMuPDFArgsDict],
) -> Union[tables.TableTransformersArgs, tables.PyMuPDFArgs]:
    if args_dict["parsing_algorithm"] == "table-transformers":
        return tables.TableTransformersArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "pymupdf":
        return tables.PyMuPDFArgs(**args_dict)
    elif args_dict["parsing_algorithm"] == "unitable":
        return tables.UnitableArgs(**args_dict)
    else:
        raise ValueError(
            f"Unsupported parsing_algorithm: {args_dict['parsing_algorithm']}"
        )


class DocumentParser:
    """
    A parser for extracting elements from PDF documents, including text and tables.

    Attributes:
        processing_pipeline (Optional[IngestionPipelineType]): A subclass of IngestionPipeline to process extracted elements.
        table_args (Optional[Union[TableTransformersArgsDict, PyMuPDFArgsDict]]): Arguments to customize table parsing.
    """

    _verbose: bool = False

    def __init__(
        self,
        *,
        processing_pipeline: Union[IngestionPipeline, NotGiven, None] = NOT_GIVEN,
        table_args: Union[
            TableTransformersArgsDict, PyMuPDFArgsDict, NotGiven
        ] = NOT_GIVEN,
    ):
        self.processing_pipeline: IngestionPipeline
        if processing_pipeline is NOT_GIVEN:
            self.processing_pipeline = BasicIngestionPipeline()
        elif processing_pipeline is None:
            self.processing_pipeline = NoOpIngestionPipeline()
        else:
            self.processing_pipeline = processing_pipeline  # type: ignore

        self.processing_pipeline.verbose = self._verbose

        self.table_args = table_args

    def parse(
        self,
        file: Union[str, Path],
        ocr: bool = False,
    ) -> ParsedDocument:
        """
        Parse a given document.

        Args:
            file (Union[str, Path]): The path to the PDF file.
            ocr (bool): Whether to use OCR for text extraction. Not recommended unless necessary - inherently slower and less accurate. Note uses PyMuPDF for OCR.
        """
        doc = Pdf(file)

        text_engine: Literal["pdfminer", "pymupdf"] = (
            "pdfminer" if not ocr else "pymupdf"
        )
        text_elems = text.ingest(doc, parsing_method=text_engine)
        text_nodes = self._elems_to_nodes(text_elems)

        table_nodes = []
        table_args_obj = None
        if self.table_args:
            table_args_obj = _table_args_dict_to_model(self.table_args)
            table_elems = tables.ingest(doc, table_args_obj, verbose=self._verbose)
            table_nodes = self._elems_to_nodes(table_elems)

        nodes = text_nodes + table_nodes
        nodes = self.processing_pipeline.run(nodes)

        parsed_doc = ParsedDocument(
            nodes=nodes,
            filename=Path(file).name,
            num_pages=doc.num_pages,
            coordinate_system=consts.COORDINATE_SYSTEM,
            table_parsing_kwargs=(
                table_args_obj.model_dump() if table_args_obj else None
            ),
            creation_date=doc.file_metadata.get("creation_date"),
            last_modified_date=doc.file_metadata.get("last_modified_date"),
            last_accessed_date=doc.file_metadata.get("last_accessed_date"),
            file_size=doc.file_metadata.get("file_size"),
        )
        return parsed_doc

    @staticmethod
    def _elems_to_nodes(
        elems: Union[List[TextElement], List[TableElement], List[ImageElement]],
    ) -> List[Node]:
        return [
            Node(
                elements=(e,),
            )
            for e in elems
        ]
