from typing import Literal, Optional, Sequence, Any, DefaultDict, TypedDict, List
from collections import defaultdict, namedtuple
from enum import Enum

from pydantic import (
    BaseModel,
    model_validator,
)

from src import consts
from src.utils import num_tokens

AggregatePosition = namedtuple("AggregatePosition", ["min_page", "min_y0", "min_x0"])


class PrevNodeSimilarity(TypedDict):
    prev_similarity: float
    node: "Node"


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

    @property
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


#####################
### TEXT ELEMENTS ###
#####################


class LineElement(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]
    _font_size: Optional[float] = None
    _font_flags: Optional[int] = None

    @property
    def is_bold(self) -> bool:
        if self._font_flags is None:
            return False
        return bool(self._font_flags & 2**4)

    @property
    def is_italic(self) -> bool:
        if self._font_flags is None:
            return False
        return bool(self._font_flags & 2**1)

    @property
    def is_heading(self) -> bool:
        MIN_HEADING_SIZE = 14
        return (
            self._font_size is not None
            and self._font_size >= MIN_HEADING_SIZE
            and self.is_bold
        )

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

    def same_level(self, other: "LineElement", error_margin: float = 0.0) -> bool:
        y_distance = abs(self.bbox[1] - other.bbox[1])

        return y_distance <= error_margin

    def combine(self, other: "LineElement") -> "LineElement":
        new_bbox = (
            min(self.bbox[0], other.bbox[0]),
            min(self.bbox[1], other.bbox[1]),
            max(self.bbox[2], other.bbox[2]),
            max(self.bbox[3], other.bbox[3]),
        )
        return LineElement(
            text=self.text + " " + other.text,
            bbox=new_bbox,
        )


class TextElement(BaseModel):
    text: str
    lines: list[LineElement]
    bbox: Bbox
    variant: Literal[NodeVariant.TEXT] = NodeVariant.TEXT

    @property
    def tokens(self) -> int:
        return num_tokens(self.text)

    @property
    def is_stub(self) -> bool:
        return self.tokens < 50

    @property
    def is_small(self) -> bool:
        return self.tokens < consts.TOKENIZATION_LOWER_LIMIT

    @property
    def is_large(self) -> bool:
        return self.tokens > consts.TOKENIZATION_UPPER_LIMIT

    @property
    def page(self) -> int:
        return self.bbox.page

    @property
    def area(self) -> float:
        return (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)

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


class Node(BaseModel):
    elements: list[TextElement]

    @property
    def tokens(self) -> int:
        return sum([e.tokens for e in self.elements])

    @property
    def is_stub(self) -> bool:
        return self.tokens < 50

    @property
    def is_small(self) -> bool:
        return self.tokens < consts.TOKENIZATION_LOWER_LIMIT

    @property
    def is_large(self) -> bool:
        return self.tokens > consts.TOKENIZATION_UPPER_LIMIT

    @property
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

    @property
    def num_pages(self) -> int:
        return len(set(element.bbox.page for element in self.elements))

    @property
    def start_page(self) -> int:
        return min(element.bbox.page for element in self.elements)

    @property
    def end_page(self) -> int:
        return max(element.bbox.page for element in self.elements)

    @property
    def text(self) -> str:
        return "\n".join([e.text for e in self.elements])

    def overlaps(
        self, other: "Node", x_error_margin: float = 0.0, y_error_margin: float = 0.0
    ) -> bool:
        # Iterate through each bounding box in the current node
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

    @property
    def aggregate_position(self) -> AggregatePosition:
        """
        Calculate an aggregate position for the node based on its elements.
        Returns a tuple of (min_page, min_y0, min_x0) to use as sort keys.
        """
        min_page = min(element.bbox.page for element in self.elements)
        min_y0 = min(element.bbox.y0 for element in self.elements)
        min_x0 = min(element.bbox.x0 for element in self.elements)
        return AggregatePosition(min_page, min_y0, min_x0)

    def combine(self, other: "Node") -> "Node":
        return Node(elements=self.elements + other.elements)


######################
### TABLE ELEMENTS ###
######################


class TableElement(BaseModel):
    text: str
    page: int
    bbox: Bbox
    variant: Literal[NodeVariant.TABLE] = NodeVariant.TABLE

    @property
    def area(self) -> float:
        return (self.bbox.x1 - self.bbox.x0) * (self.bbox.y1 - self.bbox.y0)


################
### DOCUMENT ###
################


class FileMetadata(BaseModel):
    filename: str
    num_pages: int


class ParsedDoc(BaseModel):
    nodes: List["Node"]
    file_metadata: FileMetadata
