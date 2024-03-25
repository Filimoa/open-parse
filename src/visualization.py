import random
from typing import Union, List, Sequence
from pathlib import Path

import fitz

from src.schemas import TextElement, TableElement, PrevNextSimilarity


def draw_bboxes(
    file: str | Path | fitz.Document,
    elements: Sequence[TextElement | TableElement],
    draw_sub_elements: bool = False,
) -> fitz.Document:
    if draw_sub_elements:
        raise NotImplementedError("Sub-elements are not yet supported.")

    if isinstance(file, str) or isinstance(file, Path):
        pdf = fitz.open(file)
    elif isinstance(file, fitz.Document):
        pdf = fitz.open()
        pdf.insert_pdf(file)

    else:
        raise TypeError(f"Invalid type for file: {type(file)}")
    for page in pdf:
        page.wrap_contents()

        for element in elements:
            if element.page != page.number:
                continue
            r = fitz.Rect(
                p0=element.bbox.page,
                p1=element.bbox.page,
                x0=element.bbox.x0,
                y0=element.bbox.y0,
                x1=element.bbox.x1,
                y1=element.bbox.y1,
            )
            color = (
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
            )
            page.draw_rect(r, color)
    return pdf


def display_pdf(
    file: fitz.Document,
    page_num: int,
) -> None:
    try:
        from IPython.display import Image, display  # type: ignore
    except ImportError:
        raise ImportError(
            "IPython is required to display PDFs. Please install it with `pip install ipython`."
        )
    page = file[page_num]
    img_data = page.get_pixmap().tobytes("png")
    display(Image(data=img_data))


def draw_prev_next_sim_bboxes(
    file: Union[str, Path, fitz.Document],
    elements: List[PrevNextSimilarity],
) -> fitz.Document:
    if isinstance(file, (str, Path)):
        pdf = fitz.open(file)
    elif isinstance(file, fitz.Document):
        pdf = fitz.open(stream=file.write(), filetype="pdf")
    else:
        raise TypeError(f"Invalid type for file: {type(file)}")

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
