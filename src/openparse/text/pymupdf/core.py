from typing import List

from openparse.pdf import Pdf
from openparse.schemas import Bbox, LineElement, TextElement, TextSpan


def flags_decomposer(flags: int) -> str:
    """Make font flags human readable."""
    attrs = []
    if flags & 2**0:
        attrs.append("superscript")
    if flags & 2**1:
        attrs.append("italic")
    if flags & 2**2:
        attrs.append("serifed")
    else:
        attrs.append("sans")
    if flags & 2**3:
        attrs.append("monospaced")
    else:
        attrs.append("proportional")
    if flags & 2**4:
        attrs.append("bold")
    return ", ".join(attrs)


def is_bold(flags) -> bool:
    return bool(flags & 2**4)


def is_italic(flags) -> bool:
    return bool(flags & 2**1)


def _lines_from_ocr_output(lines: dict, error_margin: float = 0) -> List[LineElement]:
    """
    Creates LineElement objects from given lines, combining overlapping ones.
    """
    combined: List[LineElement] = []

    for line in lines:
        bbox = line["bbox"]
        spans = [
            TextSpan(
                text=span["text"],
                is_bold=is_bold(span["flags"]),
                is_italic=is_italic(span["flags"]),
                size=span["size"],
            )
            for span in line["spans"]
        ]

        line_element = LineElement(bbox=bbox, spans=tuple(spans))
        for i, other in enumerate(combined):
            overlaps = line_element.overlaps(other, error_margin=error_margin)
            similar_height = line_element.is_at_similar_height(
                other, error_margin=error_margin
            )

            if overlaps and similar_height:
                combined[i] = line_element.combine(other)
                break
        else:
            combined.append(line_element)

    return combined


def ingest(
    doc: Pdf,
) -> List[TextElement]:
    """Parses text elements from a given pdf document."""
    elements = []
    pdoc = doc.to_pymupdf_doc()
    for page_num, page in enumerate(pdoc):
        page_ocr = page.get_textpage_ocr(flags=0, full=False)
        for node in page.get_text("dict", textpage=page_ocr, sort=True)["blocks"]:
            if node["type"] != 0:
                continue

            lines = _lines_from_ocr_output(node["lines"])

            # Flip y-coordinates to match the top-left origin system
            fy0 = page.rect.height - node["bbox"][3]
            fy1 = page.rect.height - node["bbox"][1]

            elements.append(
                TextElement(
                    bbox=Bbox(
                        x0=node["bbox"][0],
                        y0=fy0,
                        x1=node["bbox"][2],
                        y1=fy1,
                        page=page_num,
                        page_width=page.rect.width,
                        page_height=page.rect.height,
                    ),
                    text="\n".join(line.text for line in lines),
                    lines=tuple(lines),
                )
            )
    return elements
