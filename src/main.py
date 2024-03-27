from typing import List, Union, TypedDict, Optional, Literal
from pathlib import Path

from src import text, tables
from src.processing import run_pipeline, ProcessingStep, default_pipeline
from src.schemas import Node, TextElement, TableElement
from src.pdf import Pdf


class TableTransformersArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["table-transformers"]
    min_table_confidence: float
    min_cell_confidence: float
    table_output_format: Literal["str", "markdown", "html"]


class PyMuPDFArgsDict(TypedDict, total=False):
    parsing_algorithm: Literal["pymupdf"]
    table_output_format: Literal["str", "markdown", "html"]


def _table_args_dict_to_model(
    args_dict: Union[
        TableTransformersArgsDict,
        PyMuPDFArgsDict,
        None,
    ]
) -> Union[tables.TableTransformersArgs, tables.PyMuPDFArgs]:
    if args_dict is None:
        args_dict = PyMuPDFArgsDict()
    parsing_algorithm = args_dict.get("parsing_algorithm", "table-transformers")

    if parsing_algorithm == "table-transformers":
        return tables.TableTransformersArgs(**args_dict)
    elif parsing_algorithm == "pymupdf":
        return tables.PyMuPDFArgs(**args_dict)
    else:
        raise ValueError(f"Unsupported parsing_algorithm: {parsing_algorithm}")


class DocumentParser:
    def __init__(
        self,
        processing_pipeline: Optional[list[ProcessingStep]] = None,
        table_args: Union[TableTransformersArgsDict, PyMuPDFArgsDict, None] = None,
    ):
        if not processing_pipeline:
            processing_pipeline = default_pipeline

        self.table_args = table_args

    def parse(
        self,
        file: str | Path,
    ) -> List[Node]:
        doc = Pdf(file)

        text_elems = text.ingest(doc)
        text_nodes = self._elems_to_nodes(text_elems)

        if self.table_args:
            args_obj = _table_args_dict_to_model(self.table_args)
            table_elems = tables.ingest(doc, args_obj)
            table_nodes = self._elems_to_nodes(table_elems)

        nodes = text_nodes + table_nodes
        processed_elems = run_pipeline(nodes)
        return processed_elems

    @staticmethod
    def _elems_to_nodes(
        elems: Union[List[TextElement], List[TableElement]]
    ) -> List[Node]:
        return [
            Node(
                elements=(e,),
            )
            for e in elems
        ]
