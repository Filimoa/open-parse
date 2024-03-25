from typing import Union, List, Optional, Literal, Tuple, Any, Sequence

import fitz
from pydantic import BaseModel, model_validator

from src.tables.utils import doc_to_imgs


###############
### SCHEMAS ###
###############

Size = Tuple[int, int]
BBox = Tuple[float, float, float, float]
MIN_CONFIDENCE = 0.95


class _TableCellModelOutput(BaseModel):
    label: Literal[
        "table spanning cell",
        "table row",
        "table column",
        "table",
        "table column header",
    ]
    confidence: float
    bbox: BBox  # note: image coordinates


class _TableModelOutput(BaseModel):
    label: Literal["table",]
    confidence: float
    bbox: BBox  # note: image coordinates


class TableHeaderCell(BaseModel):
    bbox: BBox
    content: Optional[str] = None
    variant: Literal["header"] = "header"


class TableDataCell(BaseModel):
    bbox: BBox
    content: Optional[str] = None
    variant: Literal["data"] = "data"


class TableHeader(BaseModel):
    cells: List[TableHeaderCell]

    def sort_cells(self) -> None:
        self.cells.sort(key=lambda cell: (cell.bbox[1], cell.bbox[0]))

    @property
    def bbox(self) -> BBox:
        x0 = min(cell.bbox[0] for cell in self.cells)
        y0 = min(cell.bbox[1] for cell in self.cells)
        x1 = max(cell.bbox[2] for cell in self.cells)
        y1 = max(cell.bbox[3] for cell in self.cells)
        return (x0, y0, x1, y1)


class TableRow(BaseModel):
    cells: List[TableDataCell]

    def sort_cells(self) -> None:
        self.cells.sort(key=lambda cell: (cell.bbox[1], cell.bbox[0]))

    @property
    def bbox(self) -> BBox:
        x0 = min(cell.bbox[0] for cell in self.cells)
        y0 = min(cell.bbox[1] for cell in self.cells)
        x1 = max(cell.bbox[2] for cell in self.cells)
        y1 = max(cell.bbox[3] for cell in self.cells)
        return (x0, y0, x1, y1)


class Table(BaseModel):
    bbox: BBox
    headers: List[TableHeader]
    rows: List[TableRow]

    ###################
    ### TABLE UTILS ###
    ###################

    @model_validator(mode="before")
    @classmethod
    def sort_and_validate(cls, values):
        """Sort headers and rows"""
        headers = sorted(values.get("headers", []), key=lambda h: h.bbox[1])
        rows = sorted(values.get("rows", []), key=lambda r: r.bbox[1])

        for header in headers:
            header.sort_cells()

        for row in rows:
            row.sort_cells()

        values["headers"] = headers
        values["rows"] = rows
        return values

    def pprint(self) -> None:
        column_widths = self._calc_col_widths()

        self._print_horizontal_border(column_widths)

        for header in self.headers:
            self._print_row(header.cells, column_widths)
            self._print_horizontal_border(column_widths)

        for row in self.rows:
            self._print_row(row.cells, column_widths)
            self._print_horizontal_border(column_widths)

    def _calc_col_widths(self) -> List[int]:
        max_widths = [
            max(len(cell.content or "") for cell in column)
            for column in zip(
                *[header.cells for header in self.headers]
                + [row.cells for row in self.rows]
            )
        ]
        return max_widths

    def _print_row(
        self,
        cells: Sequence[Union[TableHeaderCell, TableDataCell]],
        column_widths: List[int],
    ) -> None:
        row_content = "|".join(
            " {} ".format(cell.content.ljust(width) if cell.content else " " * width)
            for cell, width in zip(cells, column_widths)
        )
        print("|{}|".format(row_content))

    def _print_horizontal_border(self, column_widths: List[int]) -> None:
        border = "+".join("-" * (width + 2) for width in column_widths)
        print("+{}+".format(border))

    def sort(self) -> None:
        self.headers.sort(
            key=lambda header: (header.cells[0].bbox[1], header.cells[0].bbox[0])
        )
        for header in self.headers:
            header.sort_cells()

        self.rows.sort(key=lambda row: (row.cells[0].bbox[1], row.cells[0].bbox[0]))
        for row in self.rows:
            row.sort_cells()

    def _run_ocr(self, pdf_page: fitz.Page):
        for header in self.headers:
            for hcell in header.cells:
                cell_rect = fitz.Rect(hcell.bbox)
                hcell.content = pdf_page.get_textbox(cell_rect)

        for row in self.rows:
            for rcell in row.cells:
                cell_rect = fitz.Rect(rcell.bbox)
                rcell.content = pdf_page.get_textbox(cell_rect)
