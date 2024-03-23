from typing import Union, List, Optional, Literal, Self, Tuple, Any, Sequence

import fitz  # type: ignore

from src.utils import _read_pdf_as_imgs
from .schemas import (
    Size,
    BBox,
    Table,
)
from .ml import find_table_bboxes, get_table_content


def parse_tables(pdf_document: fitz.Document) -> List[Table]:
    pdf_as_imgs = _read_pdf_as_imgs(pdf_document)

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
