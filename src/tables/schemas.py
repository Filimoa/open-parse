from typing import Union, List, Optional, Literal, Self, Tuple, Any, Sequence

import fitz  # type: ignore
from pydantic import BaseModel, conlist, validator, root_validator
from PIL import Image  # type: ignore

from src.utils import _read_pdf_as_imgs
from .ml import find_table_bboxes, get_table_content
from .geometry import _calc_bbox_intersection, _convert_img_cords_to_pdf_cords


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

    @root_validator(pre=True)
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

    @classmethod
    def from_model_outputs(
        cls,
        image_size: Size,
        page_size: Size,
        table_bbox: BBox,
        table_cells: List[_TableCellModelOutput],
    ) -> "Table":
        headers = [
            cell
            for cell in table_cells
            if cell.label == "table column header" and cell.confidence > MIN_CONFIDENCE
        ]
        rows = [
            cell
            for cell in table_cells
            if cell.label == "table row" and cell.confidence > MIN_CONFIDENCE
        ]
        cols = [
            cell
            for cell in table_cells
            if cell.label == "table column" and cell.confidence > MIN_CONFIDENCE
        ]

        header_objs = cls._preprocess_header_cells(headers, cols, image_size, page_size)
        row_objs = cls._process_row_cells(
            rows, cols, header_objs, image_size, page_size
        )

        return cls(bbox=table_bbox, headers=header_objs, rows=row_objs)

    @staticmethod
    def _preprocess_header_cells(
        header_rows: List[_TableCellModelOutput],
        cols: List[_TableCellModelOutput],
        image_size: Size,
        page_size: Size,
    ) -> List[TableHeader]:
        header_cells = []
        for header in header_rows:
            header_row_cells = []
            for col in cols:
                cell_bbox = _calc_bbox_intersection(
                    header.bbox, col.bbox, safety_margin=5
                )
                if cell_bbox:
                    cell_bbox = _convert_img_cords_to_pdf_cords(
                        cell_bbox, page_size, image_size
                    )
                    header_row_cells.append(
                        TableHeaderCell(
                            bbox=cell_bbox,
                        )
                    )
            header_cells.append(TableHeader(cells=header_row_cells))
        return header_cells

    @staticmethod
    def _process_row_cells(
        rows: List[_TableCellModelOutput],
        cols: List[_TableCellModelOutput],
        headers: List[TableHeader],
        image_size: Size,
        page_size: Size,
    ) -> List[TableRow]:
        """
        Process row cells by checking against header cells for overlaps and converting coordinates.
        """
        data_cells = []
        for row in rows:
            row_cells = []
            for col in cols:
                cell_bbox = _calc_bbox_intersection(row.bbox, col.bbox, safety_margin=5)
                if cell_bbox:
                    cell_bbox_pdf = _convert_img_cords_to_pdf_cords(
                        cell_bbox, page_size, image_size
                    )
                    if not Table._is_overlapping_with_headers(cell_bbox_pdf, headers):
                        row_cells.append(
                            TableDataCell(
                                bbox=cell_bbox_pdf,
                            )
                        )
            if row_cells:
                data_cells.append(TableRow(cells=row_cells))
        return data_cells

    @staticmethod
    def _is_overlapping_with_headers(
        cell_bbox: BBox, headers: List[TableHeader]
    ) -> bool:
        """
        Check if a given cell's bounding box overlaps with any of the header cells' bounding boxes.
        """
        for header in headers:
            for hcell in header.cells:
                if (
                    cell_bbox[0] < hcell.bbox[2]
                    and cell_bbox[2] > hcell.bbox[0]
                    and cell_bbox[1] < hcell.bbox[3]
                    and cell_bbox[3] > hcell.bbox[1]
                ):
                    return True
        return False
