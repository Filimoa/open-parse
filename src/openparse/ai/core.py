import datetime as dt
import uuid
from collections import defaultdict
from functools import cached_property
from typing import Any, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, computed_field, model_validator

from openparse import consts


class LineElement(BaseModel):
    bbox: Tuple[float, float, float, float]
    text: str
    style: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def round_bbox_vals(cls, data: Any) -> Any:
        data["bbox"] = tuple(round(val, 2) for val in data["bbox"])
        return data


class Node(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        exclude=True,
    )
    elements: Tuple[Union[TextElement, TableElement], ...] = Field(
        exclude=True, frozen=True
    )
    coordinate_system: Literal["top-left", "bottom-left"] = Field(
        default=consts.COORDINATE_SYSTEM, frozen=True, exclude=True
    )  # controlled globally for now, should be moved into elements
    embedding: Optional[List[float]] = Field(
        default=None, description="Embedding of the node."
    )

    @computed_field  # type: ignore
    @cached_property
    def node_id(self) -> str:
        return self.id_

    @computed_field  # type: ignore
    @cached_property
    def variant(self) -> Set[Literal["text", "table"]]:
        return {e.variant.value for e in self.elements}

    @computed_field  # type: ignore
    @cached_property
    def tokens(self) -> int:
        return sum([e.tokens for e in self.elements])

    @computed_field  # type: ignore
    @cached_property
    def bbox(self) -> List[Bbox]:
        elements_by_page = defaultdict(list)
        for element in self.elements:
            elements_by_page[element.bbox.page].append(element)

        # Calculate bounding box for each page
        bboxes = []
        for page, elements in elements_by_page.items():
            x0 = min(e.bbox.x0 for e in elements)
            y0 = min(e.bbox.y0 for e in elements)
            x1 = max(e.bbox.x1 for e in elements)
            y1 = max(e.bbox.y1 for e in elements)
            page_height = elements[0].bbox.page_height
            page_width = elements[0].bbox.page_width
            bboxes.append(
                Bbox(
                    page=page,
                    page_height=page_height,
                    page_width=page_width,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                )
            )

        return bboxes

    @cached_property
    def is_stub(self) -> bool:
        return self.tokens < 50

    @cached_property
    def is_small(self) -> bool:
        return self.tokens < self.tokenization_lower_limit

    @cached_property
    def is_large(self) -> bool:
        return self.tokens > self.tokenization_upper_limit

    @cached_property
    def num_pages(self) -> int:
        return len({element.bbox.page for element in self.elements})

    @cached_property
    def start_page(self) -> int:
        return min(element.bbox.page for element in self.elements)

    @cached_property
    def end_page(self) -> int:
        return max(element.bbox.page for element in self.elements)

    def to_llama_index(self):
        try:
            from llama_index.core.schema import TextNode as LlamaIndexTextNode
        except ImportError as err:
            raise ImportError(
                "llama_index is not installed. Please install it with `pip install llama-index`."
            ) from err
        return LlamaIndexTextNode(
            id_=self.id_,
            text=self.text,
            embedding=self.embedding,
            metadata={"bbox": [b.model_dump(mode="json") for b in self.bbox]},
            excluded_embed_metadata_keys=["bbox"],
            excluded_llm_metadata_keys=["bbox"],
        )

    def _repr_markdown_(self):
        """
        When called in a Jupyter environment, this will display the node as Markdown, which Jupyter will then render as HTML.
        """
        return self.text


#######################
### PARSED DOCUMENT ###
#######################


class ParsedDocument(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        exclude=True,
    )
    nodes: List[Node]
    filename: str
    num_pages: int
    coordinate_system: Literal["top-left", "bottom-left"] = "bottom-left"
    table_parsing_kwargs: Optional[dict] = None
    last_modified_date: Optional[dt.date] = None
    last_accessed_date: Optional[dt.date] = None
    creation_date: Optional[dt.date] = None
    file_size: Optional[int] = None
    tokenization_lower_limit: int = Field(
        default=consts.TOKENIZATION_LOWER_LIMIT, frozen=True, exclude=True
    )
    tokenization_upper_limit: int = Field(
        default=consts.TOKENIZATION_UPPER_LIMIT, frozen=True, exclude=True
    )

    @cached_property
    @computed_field
    def doc_id(self) -> str:
        return self.id_

    def to_llama_index_nodes(self):
        try:
            from llama_index.core.schema import Document as LlamaIndexDocument
        except ImportError as err:
            raise ImportError(
                "llama_index is not installed. Please install it with `pip install llama-index`."
            ) from err

        li_doc = LlamaIndexDocument(
            id_=self.id_,
            metadata={
                "file_name": self.filename,
                "file_size": self.file_size,
                "creation_date": self.creation_date.isoformat(),
                "last_modified_date": self.last_modified_date.isoformat(),
            },
            excluded_embed_metadata_keys=[
                "file_size",
                "creation_date",
                "last_modified_date",
            ],
            excluded_llm_metadata_keys=[
                "file_name",
                "file_size",
                "creation_date",
                "last_modified_date",
            ],
        )
        li_nodes = self._nodes_to_llama_index(li_doc)

        return li_nodes

    def _nodes_to_llama_index(self, llama_index_doc):
        try:
            from llama_index.core.schema import NodeRelationship
        except ImportError as err:
            raise ImportError(
                "llama_index is not installed. Please install it with `pip install llama-index`."
            ) from err

        li_nodes = [node.to_llama_index() for node in sorted(self.nodes)]
        for i in range(len(li_nodes) - 1):
            li_nodes[i].relationships[NodeRelationship.NEXT] = li_nodes[
                i + 1
            ].as_related_node_info()

            li_nodes[i + 1].relationships[NodeRelationship.PREVIOUS] = li_nodes[
                i
            ].as_related_node_info()

        for li_node in li_nodes:
            li_node.relationships[NodeRelationship.PARENT] = (
                llama_index_doc.as_related_node_info()
            )

        return li_nodes
