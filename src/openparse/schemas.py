import datetime as dt
import re
import uuid
from collections import defaultdict, namedtuple
from enum import Enum
from functools import cached_property
from typing import Any, List, Literal, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from openparse import consts
from openparse.utils import num_tokens

bullet_regex = re.compile(
    r"^(\s*[\-•](?!\*)|\s*\*(?!\*)|\s*\d+\.\s|\s*\([a-zA-Z0-9]+\)\s|\s*[a-zA-Z]\.\s)"
)

ReadingOrder = namedtuple("ReadingOrder", "min_page y_position min_x0")


class NodeVariant(Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class Bbox(BaseModel):
    page: int
    page_height: float
    page_width: float
    x0: float
    y0: float
    x1: float
    y1: float

    @cached_property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @model_validator(mode="before")
    @classmethod
    def x1_must_be_greater_than_x0(cls, data: Any) -> Any:
        if "x0" in data and data["x1"] <= data["x0"]:
            raise ValueError("x1 must be greater than x0")
        return data

    @model_validator(mode="before")
    @classmethod
    def y1_must_be_greater_than_y0(cls, data: Any) -> Any:
        if "y0" in data and data["y1"] <= data["y0"]:
            raise ValueError("y1 must be greater than y0")
        return data

    def combine(self, other: "Bbox") -> "Bbox":
        if self.page != other.page:
            raise ValueError("Bboxes must be from the same page to combine.")
        return Bbox(
            page=self.page,
            page_height=self.page_height,
            page_width=self.page_width,
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    model_config = ConfigDict(frozen=True)


#####################
### TEXT ELEMENTS ###
#####################


class TextSpan(BaseModel):
    text: str
    is_bold: bool
    is_italic: bool
    size: float

    @cached_property
    def is_heading(self) -> bool:
        MIN_HEADING_SIZE = 16
        return self.size >= MIN_HEADING_SIZE and self.is_bold

    def formatted_text(
        self,
        previous_span: Optional["TextSpan"] = None,
        next_span: Optional["TextSpan"] = None,
    ) -> str:
        """Format text considering adjacent spans to avoid redundant markdown symbols."""
        formatted = self.text

        # Check if style changes at the beginning
        if self.is_bold and (previous_span is None or not previous_span.is_bold):
            formatted = f"**{formatted}"
        if self.is_italic and (previous_span is None or not previous_span.is_italic):
            formatted = f"*{formatted}"

        # Check if style changes at the end
        if self.is_bold and (next_span is None or not next_span.is_bold):
            formatted = f"{formatted}**"
        if self.is_italic and (next_span is None or not next_span.is_italic):
            formatted = f"{formatted}*"

        return formatted

    model_config = ConfigDict(frozen=True)


class LineElement(BaseModel):
    bbox: Tuple[float, float, float, float]
    spans: Tuple[TextSpan, ...]
    style: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def round_bbox_vals(cls, data: Any) -> Any:
        data["bbox"] = tuple(round(val, 2) for val in data["bbox"])
        return data

    @computed_field  # type: ignore
    @cached_property
    def text(self) -> str:
        """
        Combine spans into a single text string, respecting markdown syntax.
        """
        if not self.spans:
            return ""

        combined_text = ""
        for i, span in enumerate(self.spans):
            previous_span = self.spans[i - 1] if i > 0 else None
            next_span = self.spans[i + 1] if i < len(self.spans) - 1 else None
            combined_text += span.formatted_text(previous_span, next_span)

        cleaned_text = self._clean_markdown_formatting(combined_text)
        return cleaned_text

    @cached_property
    def is_bold(self) -> bool:
        # ignore last span for formatting, often see weird trailing spans
        spans = self.spans[:-1] if len(self.spans) > 1 else self.spans

        return all(span.is_bold for span in spans)

    @cached_property
    def is_italic(self) -> bool:
        # ignore last span for formatting, often see weird trailing spans
        spans = self.spans[:-1] if len(self.spans) > 1 else self.spans
        return all(span.is_italic for span in spans)

    @cached_property
    def is_heading(self) -> bool:
        # ignore last span for formatting, often see weird trailing spans
        spans = self.spans[:-1] if len(self.spans) > 1 else self.spans
        MIN_HEADING_SIZE = 16
        return all(span.size >= MIN_HEADING_SIZE and span.is_bold for span in spans)

    def _clean_markdown_formatting(self, text: str) -> str:
        """
        Uses regex to clean up markdown formatting, ensuring symbols don't surround whitespace.
        This will fix issues with bold (** or __) and italic (* or _) markdown where there may be
        spaces between the markers and the text.
        """
        patterns = [
            (
                r"(\*\*|__)\s+",
                r"\1",
            ),  # Remove space after opening bold or italic marker
            (
                r"\s+(\*\*|__)",
                r"\1",
            ),  # Remove space before closing bold or italic marker
            (r"(\*|_)\s+", r"\1"),  # Remove space after opening italic marker
            (r"\s+(\*|_)", r"\1"),  # Remove space before closing italic marker
            (
                r"(\*\*|__)(\*\*|__)",
                r"\1 \2",
            ),  # Add a space between adjacent identical markers
        ]

        cleaned_text = text
        for pattern, replacement in patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)

        return cleaned_text

    def overlaps(self, other: "LineElement", error_margin: float = 0.0) -> bool:
        x_overlap = not (
            self.bbox[0] - error_margin > other.bbox[2] + error_margin
            or other.bbox[0] - error_margin > self.bbox[2] + error_margin
        )

        y_overlap = not (
            self.bbox[1] - error_margin > other.bbox[3] + error_margin
            or other.bbox[1] - error_margin > self.bbox[3] + error_margin
        )

        return x_overlap and y_overlap

    def is_at_similar_height(
        self, other: "LineElement", error_margin: float = 0.0
    ) -> bool:
        y_distance = abs(self.bbox[1] - other.bbox[1])

        return y_distance <= error_margin

    def combine(self, other: "LineElement") -> "LineElement":
        """
        Used for spans
        """
        new_bbox = (
            min(self.bbox[0], other.bbox[0]),
            min(self.bbox[1], other.bbox[1]),
            max(self.bbox[2], other.bbox[2]),
            max(self.bbox[3], other.bbox[3]),
        )
        new_spans = tuple(self.spans + other.spans)

        return LineElement(bbox=new_bbox, spans=new_spans)

    model_config = ConfigDict(frozen=True)


class TextElement(BaseModel):
    text: str
    lines: Tuple[LineElement, ...]
    bbox: Bbox
    _embed_text: Optional[str] = None
    variant: Literal[NodeVariant.TEXT] = NodeVariant.TEXT

    @computed_field  # type: ignore
    @cached_property
    def embed_text(self) -> str:
        if self._embed_text:
            return self._embed_text

        return self.text

    @cached_property
    def tokens(self) -> int:
        return num_tokens(self.text)

    @cached_property
    def is_heading(self) -> bool:
        return all(line.is_heading for line in self.lines)

    @cached_property
    def is_bold(self) -> bool:
        return all(line.is_bold for line in self.lines)

    @cached_property
    def page(self) -> int:
        return self.bbox.page

    @cached_property
    def area(self) -> float:
        return (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)

    def is_at_similar_height(
        self,
        other: Union["TableElement", "TextElement", "ImageElement"],
        error_margin: float = 1,
    ) -> bool:
        y_distance = abs(self.bbox.y1 - other.bbox.y1)

        return y_distance <= error_margin

    def overlaps(
        self,
        other: "TextElement",
        x_error_margin: float = 0.0,
        y_error_margin: float = 0.0,
    ) -> bool:
        if self.page != other.page:
            return False
        x_overlap = not (
            self.bbox.x0 - x_error_margin > other.bbox.x1 + x_error_margin
            or other.bbox.x0 - x_error_margin > self.bbox.x1 + x_error_margin
        )
        y_overlap = not (
            self.bbox.y0 - y_error_margin > other.bbox.y1 + y_error_margin
            or other.bbox.y0 - y_error_margin > self.bbox.y1 + y_error_margin
        )

        return x_overlap and y_overlap

    model_config = ConfigDict(frozen=True)


######################
### TABLE ELEMENTS ###
######################


class TableElement(BaseModel):
    text: str
    bbox: Bbox
    _embed_text: Optional[str] = None
    variant: Literal[NodeVariant.TABLE] = NodeVariant.TABLE

    @computed_field  # type: ignore
    @cached_property
    def embed_text(self) -> str:
        if self._embed_text:
            return self._embed_text

        return self.text

    @cached_property
    def area(self) -> float:
        return (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)

    @cached_property
    def page(self) -> int:
        return self.bbox.page

    @cached_property
    def tokens(self) -> int:
        return num_tokens(self.text)

    def is_at_similar_height(
        self,
        other: Union["TableElement", "TextElement", "ImageElement"],
        error_margin: float = 1,
    ) -> bool:
        y_distance = abs(self.bbox.y1 - other.bbox.y1)

        return y_distance <= error_margin


######################
### IMAGE ELEMENTS ###
######################


class ImageElement(BaseModel):
    text: str
    bbox: Bbox
    image: str  # base64 encoded image
    image_mimetype: Union[
        Literal[
            "image/jpeg",
            "image/png",
            "image/bmp",
            "image/jbig2",
            "image/webp",
            "unknown",
        ],
        str,
    ]
    _embed_text: Optional[str] = None
    variant: Literal[NodeVariant.IMAGE] = NodeVariant.IMAGE

    model_config = ConfigDict(frozen=True)

    @computed_field  # type: ignore
    @cached_property
    def embed_text(self) -> str:
        if self._embed_text:
            return self._embed_text
        return self.text

    @cached_property
    def area(self) -> float:
        return (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)

    @cached_property
    def page(self) -> int:
        return self.bbox.page

    @cached_property
    def tokens(self) -> int:
        return 512  # Placeholder for image tokenization

    def is_at_similar_height(
        self,
        other: Union["TableElement", "TextElement", "ImageElement"],
        error_margin: float = 1,
    ) -> bool:
        y_distance = abs(self.bbox.y1 - other.bbox.y1)
        return y_distance <= error_margin

    def overlaps(self, other: "ImageElement", buffer: float = 1.0) -> bool:
        """Check if this image overlaps or is adjacent to another image, considering a buffer."""
        if self.bbox.page != other.bbox.page:
            return False

        return not (
            self.bbox.x1 + buffer < other.bbox.x0 - buffer
            or self.bbox.x0 - buffer > other.bbox.x1 + buffer
            or self.bbox.y1 + buffer < other.bbox.y0 - buffer
            or self.bbox.y0 - buffer > other.bbox.y1 + buffer
        )


#############
### NODES ###
#############


def _determine_relationship(
    elem1: Union["TextElement", "TableElement"],
    elem2: Union["TextElement", "TableElement"],
    line_threshold: float = 1,
    paragraph_threshold: float = 12,
) -> Literal["same-line", "same-paragraph", None]:
    """
    Determines the relationship between two elements (either TextElement or TableElement).
    Returns 'same-line', 'same-paragraph', or None.
    Tables are considered to have no direct relationship with other elements (None).
    """
    if isinstance(elem1, TableElement) or isinstance(elem2, TableElement):
        return None

    vertical_distance = abs(elem1.bbox.y0 - elem2.bbox.y0)

    if vertical_distance <= line_threshold:
        return "same-line"
    elif vertical_distance <= paragraph_threshold:
        return "same-paragraph"
    else:
        return None


class Node(BaseModel):
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        exclude=True,
    )
    elements: Tuple[Union[TextElement, TableElement, ImageElement], ...] = Field(
        exclude=True, frozen=True
    )
    tokenization_lower_limit: int = Field(
        default=consts.TOKENIZATION_LOWER_LIMIT, frozen=True, exclude=True
    )
    tokenization_upper_limit: int = Field(
        default=consts.TOKENIZATION_UPPER_LIMIT, frozen=True, exclude=True
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
    def variant(self) -> Set[Literal["text", "table", "image"]]:
        return {e.variant.value for e in self.elements}

    @computed_field  # type: ignore
    @cached_property
    def tokens(self) -> int:
        return sum([e.tokens for e in self.elements])

    @computed_field  # type: ignore
    @cached_property
    def images(self) -> List[ImageElement]:
        return [e for e in self.elements if e.variant == NodeVariant.IMAGE]

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

    @computed_field  # type: ignore
    @cached_property
    def text(self) -> str:
        sorted_elements = sorted(
            self.elements, key=lambda e: (e.page, -e.bbox.y1, e.bbox.x0)
        )

        texts = []
        for i in range(len(sorted_elements)):
            current = sorted_elements[i]
            if i > 0:
                previous = sorted_elements[i - 1]
                relationship = _determine_relationship(previous, current)

                if relationship == "same-line":
                    join_str = " "
                elif relationship == "same-paragraph":
                    join_str = "\n"
                else:
                    join_str = consts.ELEMENT_DELIMETER

                texts.append(join_str)

            texts.append(current.embed_text)

        return "".join(texts)

    @cached_property
    def is_heading(self) -> bool:
        if self.variant != {"text"}:
            return False
        if not self.is_stub:
            return False

        return all(element.is_heading or element.is_bold for element in self.elements)  # type: ignore

    @cached_property
    def starts_with_heading(self) -> bool:
        if not self.variant == {"text"}:
            return False
        return self.elements[0].is_heading  # type: ignore

    @cached_property
    def starts_with_bullet(self) -> bool:
        first_line = self.text.split(consts.ELEMENT_DELIMETER)[0].strip()
        if not first_line:
            return False
        return bool(bullet_regex.match(first_line))

    @cached_property
    def ends_with_bullet(self) -> bool:
        last_line = self.text.split(consts.ELEMENT_DELIMETER)[-1].strip()
        if not last_line:
            return False
        return bool(bullet_regex.match(last_line))

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

    @cached_property
    def reading_order(self) -> ReadingOrder:
        """
        To sort nodes based on their reading order, we need to calculate an aggregate position for the node. This allows us to:

        nodes = sorted(nodes, key=lambda x: x.reading_order)

        Returns a tuple of (min_page, y_position, min_x0) to use as sort keys, where y_position is adjusted based on the coordinate system.
        """
        min_page = min(element.bbox.page for element in self.elements)
        min_x0 = min(element.bbox.x0 for element in self.elements)

        if self.coordinate_system == "bottom-left":
            y_position = -min(element.bbox.y0 for element in self.elements)
        else:
            raise NotImplementedError(
                "Only 'bottom-left' coordinate system is supported."
            )

        return ReadingOrder(min_page=min_page, y_position=y_position, min_x0=min_x0)

    def overlaps(
        self, other: "Node", x_error_margin: float = 0.0, y_error_margin: float = 0.0
    ) -> bool:
        for bbox in self.bbox:
            other_bboxes = [
                other_bbox for other_bbox in other.bbox if other_bbox.page == bbox.page
            ]

            for other_bbox in other_bboxes:
                x_overlap = not (
                    bbox.x0 - x_error_margin > other_bbox.x1 + x_error_margin
                    or other_bbox.x0 - x_error_margin > bbox.x1 + x_error_margin
                )

                y_overlap = not (
                    bbox.y0 - y_error_margin > other_bbox.y1 + y_error_margin
                    or other_bbox.y0 - y_error_margin > bbox.y1 + y_error_margin
                )

                if x_overlap and y_overlap:
                    return True

        return False

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

    def __lt__(self, other: "Node") -> bool:
        if not isinstance(other, Node):
            return NotImplemented

        assert (
            self.coordinate_system == other.coordinate_system
        ), "Coordinate systems must match."

        return self.reading_order < other.reading_order

    def _repr_markdown_(self):
        """
        When called in a Jupyter environment, this will display the node as Markdown, which Jupyter will then render as HTML.
        """
        markdown_parts = []
        for element in self.elements:
            if element.variant == NodeVariant.TEXT:
                markdown_parts.append(element.text)
            elif element.variant == NodeVariant.IMAGE:
                image_data = element.image
                mime_type = element.image_mimetype
                if mime_type == "unknown":
                    mime_type = "image/png"
                markdown_image = f"![Image](data:{mime_type};base64,{image_data})"
                markdown_parts.append(markdown_image)
            elif element.variant == NodeVariant.TABLE:
                markdown_parts.append(element.text)
        return "\n\n".join(markdown_parts)

    def __add__(self, other: "Node") -> "Node":
        """
        Allows two Node instances to be combined using the '+' operator.
        The combined Node instance will contain elements from both nodes.
        """
        if not isinstance(other, Node):
            return NotImplemented()

        new_elems = self.elements + other.elements
        return Node(elements=new_elems)


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
