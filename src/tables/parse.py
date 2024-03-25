from typing import List, Literal

import fitz

from src.tables.utils import doc_to_imgs
from src.schemas import TableElement, Bbox


def ingest(
    doc: fitz.Document, table_format: Literal["str", "markdown", "html"] = "str"
) -> List[TableElement]:
    try:
        from .ml import find_table_bboxes, get_table_content
    except ImportError as e:
        raise ImportError(
            "Table detection and extraction requires the `torch`, `torchvision` and `transformers` libraries to be installed."
        )
    pdf_as_imgs = doc_to_imgs(doc)

    pages_with_tables = {}
    for page_num, img in enumerate(pdf_as_imgs):
        pages_with_tables[page_num] = find_table_bboxes(img)

    tables = []
    for page_num, table_bboxes in pages_with_tables.items():
        page = doc[page_num]
        page_dims = (page.rect.width, page.rect.height)
        for table_bbox in table_bboxes:
            table = get_table_content(page_dims, img, table_bbox.bbox)
            table._run_ocr(page)

            if table_format == "str":
                table_text = table.to_str()
            elif table_format == "markdown":
                table_text = table.to_markdown_str()
            elif table_format == "html":
                table_text = table.to_html_str()

            tables.append(
                TableElement(
                    bbox=Bbox(
                        page=page_num,
                        x0=table_bbox.bbox[0],
                        y0=table_bbox.bbox[1],
                        x1=table_bbox.bbox[2],
                        y1=table_bbox.bbox[3],
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    ),
                    text=table_text,
                )
            )

    return tables
