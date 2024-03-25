import random
from typing import Union, List, Sequence, Optional
from pathlib import Path

import fitz

from src.utils import load_doc
from src.schemas import Node, PrevNodeSimilarity


def draw_bboxes(
    file: str | Path | fitz.Document,
    nodes: list[Node],
    draw_sub_elements: bool = False,
) -> fitz.Document:
    if draw_sub_elements:
        raise NotImplementedError("Sub-elements are not yet supported.")

    pdf = load_doc(file)
    flattened_bboxes = [bbox for node in nodes for bbox in node.bbox]

    for page in pdf:
        page.wrap_contents()

        for bbox in flattened_bboxes:
            if bbox.page != page.number:
                continue
            r = fitz.Rect(
                p0=bbox.page,
                p1=bbox.page,
                x0=bbox.x0,
                y0=bbox.y0,
                x1=bbox.x1,
                y1=bbox.y1,
            )
            color = (
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
            )
            page.draw_rect(r, color)
    return pdf


def display_doc(
    doc: fitz.Document,
    page_nums: Optional[list[int]] = None,
) -> None:
    """
    Display a single page of a PDF file using IPython.
    """
    try:
        from IPython.display import Image, display  # type: ignore
    except ImportError:
        raise ImportError(
            "IPython is required to display PDFs. Please install it with `pip install ipython`."
        )

    if not page_nums:
        page_nums = list(range(doc.page_count))
    for page_num in page_nums:
        page = doc[page_num]
        img_data = page.get_pixmap().tobytes("png")
        display(Image(data=img_data))


def draw_prev_next_sim_bboxes(
    file: Union[str, Path, fitz.Document],
    elements: List[PrevNodeSimilarity],
) -> fitz.Document:
    pdf = load_doc(file)

    for page in pdf:
        page.wrap_contents()

        for node_info in elements:
            node = node_info["node"]
            prev_similarity = node_info.get("prev_similarity", 0.0)

            if node.bbox[0].page != page.number:
                continue

            r = fitz.Rect(
                node.bbox[0].x0, node.bbox[0].y0, node.bbox[0].x1, node.bbox[0].y1
            )

            color = (random.random(), random.random(), random.random())

            page.draw_rect(r, color, width=1.5)

            if not prev_similarity:
                continue
            sim_text = f"Prev Sim: {prev_similarity:.2f}"
            page.insert_text(
                (node.bbox[0].x0, node.bbox[0].y0), sim_text, fontsize=11, color=color
            )

    return pdf
