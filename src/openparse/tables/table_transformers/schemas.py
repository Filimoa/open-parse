from typing import Literal, Tuple

from pydantic import BaseModel

###############
### SCHEMAS ###
###############

Size = Tuple[int, int]
BBox = Tuple[float, float, float, float]


class _TableCellModelOutput(BaseModel):
    label: Literal[
        "table spanning cell",
        "table row",
        "table column",
        "table",
        "table column header",
        "table projected row header",  # WHAT IS THIS
    ]
    confidence: float
    bbox: BBox  # note: image coordinates

    @property
    def is_header(self) -> bool:
        return self.label in ["table column header", "table projected row header"]

    @property
    def is_row(self) -> bool:
        return self.label in ["table row"]

    @property
    def is_column(self) -> bool:
        return self.label in ["table column"]


class _TableModelOutput(BaseModel):
    label: Literal["table", "table rotated"]
    confidence: float
    bbox: BBox  # note: image coordinates
