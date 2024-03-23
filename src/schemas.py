from typing import Literal, Optional, Sequence, Any, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
from PIL import Image
import hashlib

import textwrap
from pydantic import BaseModel, field_validator, ValidationInfo

from src.dependencies import openai
from src import consts


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


class FileMetadata(BaseModel):
    filename: str
    num_pages: int


class PageMetadata(BaseModel):
    page_height: int
    page_width: int
    page_num: int
    page_label: Optional[int] = None
    metadata: Optional[str] = None

    @field_validator("page_height", "page_width", mode="before")
    @classmethod
    def coerce_to_int(cls, v: str, info: ValidationInfo) -> int:
        return int(v)


@dataclass
class LineElement:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    variant: Literal[NodeVariant.TEXT] = NodeVariant.TEXT

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    def overlaps(self, other: "LineElement", error_margin: float = 0.0) -> bool:
        x_overlap = not (
            self.x0 - error_margin > other.x1 + error_margin
            or other.x0 - error_margin > self.x1 + error_margin
        )
        return x_overlap

    def same_level(self, other: "LineElement", error_margin: float = 0.0) -> bool:
        y_distance = abs(self.y0 - other.y0)
        return y_distance <= error_margin

    def combine(self, other: "LineElement") -> "LineElement":
        return LineElement(
            text=other.text + " " + self.text,
            x0=min(self.x0, other.x0),
            x1=max(self.x1, other.x1),
            y0=self.y0,
            y1=self.y1,
        )


@dataclass
class TextElement:
    text: str
    lines: list[LineElement]
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    position: float
    tokens: int = field(init=False)
    variant: Literal[NodeVariant.TEXT] = NodeVariant.TEXT

    def __post_init__(self) -> None:
        assert self.x0 <= self.x1, "x0 must be less than x1"
        assert self.y0 <= self.y1, "y0 must be less than y1"
        self.tokens = openai.num_tokens(self.text)

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
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def is_bullet(self) -> bool:
        if not self.text:
            return False
        if self.text.startswith("•"):
            return True

        words = self.text.split()
        if not words:
            return False

        first_word = words[0]
        if first_word[-1] == ")":
            return True
        if first_word[0:-1].isdigit() and first_word[-1] == ".":
            return True
        return False

    def overlaps(
        self,
        other: "TextElement",
        x_error_margin: float = 0.0,
        y_error_margin: float = 0.0,
    ) -> bool:
        if self.page != other.page:
            return False
        x_overlap = not (
            self.x0 - x_error_margin > other.x1 + x_error_margin
            or other.x0 - x_error_margin > self.x1 + x_error_margin
        )
        y_overlap = not (
            self.y0 - y_error_margin > other.y1 + y_error_margin
            or other.y0 - y_error_margin > self.y1 + y_error_margin
        )

        return x_overlap and y_overlap

    def combine(self, other: "TextElement") -> "TextElement":
        return TextElement(
            page=self.page,
            lines=self.lines + other.lines,
            position=min(self.position, other.position),
            text=self.text + "\n" + other.text,
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
            variant=NodeVariant.TEXT,
        )

    def split(self, split_points: list[float]) -> list["TextElement"]:
        sorted_points = sorted(split_points + [self.y1])
        res = []
        current_y0 = self.y0
        for point in sorted_points:
            split_lines = [
                line for line in self.lines if current_y0 <= line.y0 <= line.y1 < point
            ]
            if split_lines:
                split_text = "\n".join(line.text for line in split_lines)
                if (
                    openai.num_tokens(split_text) >= consts.TOKENIZATION_LOWER_LIMIT
                    or point == sorted_points[-1]
                ):
                    split_element = TextElement(
                        text=split_text,
                        lines=split_lines,
                        page=self.page,
                        x0=self.x0,
                        y0=current_y0 + 1,
                        x1=self.x1,
                        y1=point - 1,
                        position=self.position,
                        variant=self.variant,
                    )
                    res.append(split_element)
                    current_y0 = point

        return res

    def neighbors(self, other: "TextElement", distance: int = 1) -> bool:
        if self.page != other.page:
            return False
        if abs(self.position - other.position) <= distance:
            return True
        return False


@dataclass
class TableElement:
    text: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    position: float
    image: Image.Image
    variant: Literal[NodeVariant.TABLE] = NodeVariant.TABLE
    tokens: int = field(init=False)

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    @property
    def image_hash(self) -> str:
        img_bytes = self.image.tobytes()

        hash_obj = hashlib.new("MD5")
        hash_obj.update(img_bytes)

        return hash_obj.hexdigest()

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    def __post_init__(self) -> None:
        assert self.x0 <= self.x1, "x0 must be less than x1"
        assert self.y0 <= self.y1, "y0 must be less than y1"
        self.tokens = openai.num_tokens(self.text)


class ParsedDoc(BaseModel):
    elements: Sequence[TextElement | TableElement]
    page_metadata: Sequence[PageMetadata]
    file_metadata: FileMetadata
