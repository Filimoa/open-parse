from typing import List, Union
from pathlib import Path

import fitz

from src.utils import load_doc
from src import text, tables
from src.schemas import Node, TextElement, TableElement

# def get_element_variants(elements: List[Union[TextElement, TableElement]]) ->


def elements_to_nodes(elements: List[Union[TextElement, TableElement]]) -> List[Node]:

    raise NotImplementedError


class PipelineStep:
    pass


class DocumentParse:
    def __init__(
        self,
        file: str | Path | fitz.Document,
        processing: List[PipelineStep],
        postprocessing: List[PipelineStep],
        parse_tables: bool = True,
    ):
        doc = load_doc(file)

        self.text_elems = text.ingest(doc)
        if parse_tables:
            self.table_elems = tables.ingest(doc)

        # Parse images (optional)

        # combine elements?

        #

        pass
