from typing import List, Union, TypedDict
from pathlib import Path

from src import text, tables
from src.processing import run_pipeline, ProcessingArgs
from src.schemas import Node
from src.pdf import Pdf


class TableArgs(TypedDict, total=False):
    parse: bool
    args: tables.TableTransformersArgs


class DocumentParser:
    def __init__(
        self,
        processing_args: Union[ProcessingArgs, None] = None,
        post_processing_args: Union[dict, None] = None,
        table_args: Union[TableArgs, None] = None,
    ):

        self.processing_args = processing_args
        self.post_processing_args = post_processing_args
        self.table_args = table_args

    def parse(
        self,
        file: str | Path,
    ) -> List[Node]:
        doc = Pdf(file)

        text_elems = text.ingest(doc)
        text_nodes = [
            Node(
                elements=[e],
            )
            for e in text_elems
        ]

        if self.table_args:
            if self.table_args["parse"]:
                table_elems = tables.ingest(doc, self.table_args["args"])

        all_elems = text_elems + table_elems
        processed_elems = run_pipeline(all_elems, self.processing_args)
        return processed_elems
