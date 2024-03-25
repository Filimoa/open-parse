from typing import List
import fitz

from src.schemas import Node, TextElement, LineElement, Bbox


def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2**0:
        l.append("superscript")
    if flags & 2**1:
        l.append("italic")
    if flags & 2**2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2**3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2**4:
        l.append("bold")
    return ", ".join(l)


def _lines_from_ocr_output(lines: dict, error_margin: float = 0) -> list[LineElement]:
    """
    Creates LineElement objects from given lines, combining overlapping ones.
    """
    combined: list[LineElement] = []

    for line in lines:
        bbox = line["bbox"]
        text = " ".join(span["text"] for span in line["spans"])

        line_element = LineElement(
            bbox=bbox,
            text=text,
        )
        for i, other in enumerate(combined):
            if line_element.overlaps(
                other, error_margin=error_margin
            ) and line_element.same_level(other, error_margin=error_margin):
                combined[i] = line_element.combine(other)
                break
        else:
            combined.append(line_element)

    return combined


def parse(pdf: fitz.Document) -> List[Node]:
    """Parses text elements from a given pdf document."""
    elements = []
    for page_num, page in enumerate(pdf):
        page_ocr = page.get_textpage_ocr(flags=0, full=False)
        for node in page.get_text("dict", textpage=page_ocr, sort=True)["blocks"]:
            if node["type"] != 0:
                continue

            lines = _lines_from_ocr_output(node["lines"])

            elements.append(
                TextElement(
                    bbox=Bbox(
                        x0=node["bbox"][0],
                        y0=node["bbox"][1],
                        x1=node["bbox"][2],
                        y1=node["bbox"][3],
                        page=page_num,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    ),
                    text="\n".join(line.text for line in lines),
                    lines=lines,
                )
            )
    return [Node(elements=[e]) for e in elements]
