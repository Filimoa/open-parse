from pathlib import Path
from typing import List, Literal, Optional, TypedDict, Union

from openparse import tables, text
from openparse.pdf import Pdf
from openparse.processing import ProcessingStep, default_pipeline, run_pipeline
from openparse.schemas import Node, TableElement, TextElement


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
    ],
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
        processing_pipeline: Optional[List[ProcessingStep]] = None,
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

        table_nodes = []
        if self.table_args:
            args_obj = _table_args_dict_to_model(self.table_args)
            table_elems = tables.ingest(doc, args_obj)
            table_nodes = self._elems_to_nodes(table_elems)

        nodes = text_nodes + table_nodes
        processed_elems = run_pipeline(nodes)
        return processed_elems

    @staticmethod
    def _elems_to_nodes(
        elems: Union[List[TextElement], List[TableElement]],
    ) -> List[Node]:
        return [
            Node(
                elements=(e,),
            )
            for e in elems
        ]
