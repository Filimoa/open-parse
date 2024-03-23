from typing import Union, List, Sequence
from pathlib import Path
import random

from src.dependencies.fitz import fitz
from src.schemas import TextElement, TableElement


def draw_bboxes(
    file: str | Path | fitz.Document,
    elements: Sequence[TextElement | TableElement],
) -> fitz.Document:
    if isinstance(file, str) or isinstance(file, Path):
        pdf = fitz.open(file)
    elif isinstance(file, fitz.Document):
        pdf = file
    else:
        raise TypeError(f"Invalid type for file: {type(file)}")
    for page in pdf:
        page.wrap_contents()

        for element in elements:
            if element.page != page.number:
                continue
            r = fitz.Rect(element.bounds)
            color = (
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
                random.randint(0, 255) / 256,
            )
            page.draw_rect(r, color)
