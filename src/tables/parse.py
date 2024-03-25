from typing import Union, List, Any, Sequence

import fitz

from src.tables.utils import doc_to_imgs
from src.tables.geometry import _calc_bbox_intersection, _convert_img_cords_to_pdf_cords
from .schemas import (
    Size,
    BBox,
    Table,
    _TableCellModelOutput,
    TableHeader,
    TableRow,
    TableHeaderCell,
    TableDataCell,
)

MIN_CONFIDENCE = 0.95


def from_model_outputs(
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

    header_objs = _preprocess_header_cells(headers, cols, image_size, page_size)
    row_objs = _process_row_cells(rows, cols, header_objs, image_size, page_size)

    return Table(bbox=table_bbox, headers=header_objs, rows=row_objs)


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
            cell_bbox = _calc_bbox_intersection(header.bbox, col.bbox, safety_margin=5)
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
                if not _is_overlapping_with_headers(cell_bbox_pdf, headers):
                    row_cells.append(
                        TableDataCell(
                            bbox=cell_bbox_pdf,
                        )
                    )
        if row_cells:
            data_cells.append(TableRow(cells=row_cells))
    return data_cells


def _is_overlapping_with_headers(cell_bbox: BBox, headers: List[TableHeader]) -> bool:
    """
    Some rows are also headers, we need to drop these. Check if a given cell's bounding box overlaps with any of the header cells' bounding boxes.
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


def parse(pdf_document: fitz.Document) -> List[Table]:
    try:
        from .ml import find_table_bboxes, get_table_content
    except ImportError:
        raise ImportError(
            "Table detection and extraction requires the `torch` and `transformers` libraries to be installed."
        )
    pdf_as_imgs = doc_to_imgs(pdf_document)

    pages_with_tables = {}
    for page_num, img in enumerate(pdf_as_imgs):
        pages_with_tables[page_num] = find_table_bboxes(img)

    tables = []
    for page_num, table_bboxes in pages_with_tables.items():
        page = pdf_document[page_num]
        page_dims = (page.rect.width, page.rect.height)
        for table_bbox in table_bboxes:
            table = get_table_content(page_dims, img, table_bbox.bbox)
            table._run_ocr(page)
            table.pprint()
            tables.append(table)

    return tables
