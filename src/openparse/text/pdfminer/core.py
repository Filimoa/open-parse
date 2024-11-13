import base64
import logging
from io import BytesIO
from typing import Any, Iterable, List, Optional, Tuple, Union

from pdfminer.layout import (
    LTAnno,
    LTChar,
    LTFigure,
    LTImage,
    LTTextContainer,
    LTTextLine,
)
from pdfminer.pdftypes import resolve1
from pdfminer.psparser import PSLiteral
from PIL import Image
from pydantic import BaseModel, model_validator

from openparse.pdf import Pdf
from openparse.schemas import Bbox, ImageElement, LineElement, TextElement, TextSpan


class CharElement(BaseModel):
    text: str
    fontname: str
    size: float

    @property
    def is_bold(self) -> bool:
        return "Bold" in self.fontname or "bold" in self.fontname

    @property
    def is_italic(self) -> bool:
        return "Italic" in self.fontname or "italic" in self.fontname

    @model_validator(mode="before")
    @classmethod
    def round_size(cls, data: Any) -> Any:
        data["size"] = round(data["size"], 2)
        return data


def _extract_chars(text_line: LTTextLine) -> List[CharElement]:
    """
    The last_fontname variable is used to keep track of the most recent fontname seen as the function iterates through text_line.

    This is necessary because LTAnno elements (annotations) do not have their own font and size information; they use the most recently encountered fontname and size from a LTChar element.
    """

    chars = []
    # take the first LTChar's fontname and size for any LTAnno before them
    last_fontname = next(
        (char.fontname for char in text_line if isinstance(char, LTChar)), ""
    )
    last_size = next((char.size for char in text_line if isinstance(char, LTChar)), 0.0)

    for char in text_line:
        if not isinstance(char, LTChar) and not isinstance(char, LTAnno):
            continue
        if isinstance(char, LTChar):
            last_fontname = char.fontname
            last_size = char.size
        chars.append(
            CharElement(text=char.get_text(), fontname=last_fontname, size=last_size)
        )

    return chars


def _get_mime_type(pdf_object: LTImage) -> Optional[str]:
    """Determine the MIME type of an image in a PDF based on its filters."""
    # Resolve the stream attributes
    stream_attrs = pdf_object.stream.attrs
    subtype = resolve1(stream_attrs.get("Subtype", ""))
    filters = resolve1(stream_attrs.get("Filter", ""))

    if isinstance(filters, list):
        filter_names = [str(f).lstrip("/").strip("\"'") for f in filters]
    else:
        filter_names = [str(filters).lstrip("/").strip("\"'")] if filters else []

    mime_type = None
    if "DCTDecode" in filter_names:
        mime_type = "image/jpeg"
    elif "JPXDecode" in filter_names:
        mime_type = "image/jp2"
    elif "FlateDecode" in filter_names:
        # Could be PNG or TIFF; may need additional information
        # For simplicity, assume PNG
        mime_type = "image/png"
    elif "CCITTFaxDecode" in filter_names:
        mime_type = "image/tiff"
    elif "JBIG2Decode" in filter_names:
        mime_type = "image/jbig2"

    return mime_type


def _group_chars_into_spans(chars: Iterable[CharElement]) -> List[TextSpan]:
    spans = []
    current_text = ""
    current_style = (False, False, 0.0)

    for char in chars:
        char_style = (char.is_bold, char.is_italic, char.size)
        # If the current character is a space, compress multiple spaces and continue loop.
        if char.text.isspace():
            if not current_text.endswith(" "):
                current_text += " "
            continue

        # If style changes and there's accumulated text, add it to spans.
        if char_style != current_style and current_text:
            # Ensure there is at most one space at the end of the text.
            spans.append(
                TextSpan(
                    text=current_text.rstrip()
                    + (" " if current_text.endswith(" ") else ""),
                    is_bold=current_style[0],
                    is_italic=current_style[1],
                    size=current_style[2],
                )
            )
            current_text = char.text
        else:
            current_text += char.text
        current_style = char_style

    # After the loop, add any remaining text as a new span.
    if current_text:
        spans.append(
            TextSpan(
                text=current_text.rstrip()
                + (" " if current_text.endswith(" ") else ""),
                is_bold=current_style[0],
                is_italic=current_style[1],
                size=current_style[2],
            )
        )
    return spans


def _create_line_element(text_line: LTTextLine) -> LineElement:
    """Create a LineElement from a text line."""
    chars = _extract_chars(text_line)
    spans = _group_chars_into_spans(chars)
    bbox = (text_line.x0, text_line.y0, text_line.x1, text_line.y1)
    return LineElement(bbox=bbox, spans=tuple(spans))


def _get_bbox(lines: List[LineElement]) -> Tuple[float, float, float, float]:
    """Get the bounding box of a list of LineElements."""
    x0 = min(line.bbox[0] for line in lines)
    y0 = min(line.bbox[1] for line in lines)
    x1 = max(line.bbox[2] for line in lines)
    y1 = max(line.bbox[3] for line in lines)
    return x0, y0, x1, y1


def _process_png_image(e: LTImage, page_rotation: int = 0) -> Optional[bytes]:
    try:
        # Extract image attributes
        width = e.stream.attrs.get("Width", 0)
        height = e.stream.attrs.get("Height", 0)
        color_space = e.stream.attrs.get("ColorSpace", "DeviceRGB")
        decode = e.stream.attrs.get("Decode", None)

        # Resolve indirect references
        color_space = resolve1(color_space)
        decode = resolve1(decode)

        # Ensure color_space is a string
        if isinstance(color_space, list):
            color_space = color_space[0]
        if isinstance(color_space, PSLiteral):
            color_space = color_space.name
        if isinstance(color_space, str):
            color_space = color_space.strip("/")
        else:
            logging.info(f"Unsupported color space type: {type(color_space)}")
            return None

        # Map PDF color space to PIL mode
        if color_space == "DeviceRGB":
            mode = "RGB"
        elif color_space == "DeviceGray":
            mode = "L"
        elif color_space == "DeviceCMYK":
            mode = "CMYK"
        else:
            logging.info(f"Unsupported color space: {color_space}")
            return None

        # Get the image data after filters have been applied
        img_data = e.stream.get_data()

        # Create image using Pillow
        img = Image.frombytes(mode, (width, height), img_data)

        # Convert to PNG bytes
        output = BytesIO()
        img.save(output, format="PNG")
        return output.getvalue()
    except Exception as ex:
        logging.error(f"Error processing PNG image: {ex}")
        return None


def ingest(pdf_input: Pdf) -> List[Union[TextElement, ImageElement]]:
    """Parse PDF and return a list of TextElement and ImageElement objects."""
    elements = []
    page_layouts = pdf_input.extract_layout_pages()

    for page_num, page_layout in enumerate(page_layouts):
        page_width = page_layout.width
        page_height = page_layout.height
        page_elements = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                lines = []
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        lines.append(_create_line_element(text_line))
                if not lines:
                    continue
                bbox = _get_bbox(lines)

                page_elements.append(
                    TextElement(
                        bbox=Bbox(
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            page=page_num,
                            page_width=page_width,
                            page_height=page_height,
                        ),
                        text="\n".join(line.text for line in lines),
                        lines=tuple(lines),
                    )
                )
            elif isinstance(element, LTFigure):
                for e in element:
                    if isinstance(e, LTImage):
                        mime_type = _get_mime_type(e)
                        if mime_type:
                            if mime_type == "image/png":
                                img_data = _process_png_image(e)
                            else:
                                img_data = e.stream.get_data()
                            if img_data:
                                base64_string = base64.b64encode(img_data).decode(
                                    "utf-8"
                                )
                                page_elements.append(
                                    ImageElement(
                                        bbox=Bbox(
                                            x0=e.bbox[0],
                                            y0=e.bbox[1],
                                            x1=e.bbox[2],
                                            y1=e.bbox[3],
                                            page=page_num,
                                            page_width=page_width,
                                            page_height=page_height,
                                        ),
                                        image=base64_string,
                                        image_mimetype=mime_type or "unknown",
                                        text="",
                                    )
                                )
        elements.extend(page_elements)
    return elements
