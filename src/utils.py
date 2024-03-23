import logging
from typing import Union, List, Tuple, Any

import fitz  # type: ignore
from PIL import Image  # type: ignore


Size = Tuple[int, int]
BBox = Tuple[float, float, float, float]


###################
### IMAGE UTILS ###
###################


def _crop_img_with_padding(
    image: Image.Image, bbox: BBox, padding: int = 10
) -> Image.Image:
    if padding < 0:
        raise ValueError("Padding must be non-negative")

    left, top, right, bottom = bbox
    if not (0 <= left < right <= image.width) or not (
        0 <= top < bottom <= image.height
    ):
        raise ValueError("Bounding box is out of the image boundaries")

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    try:
        return image.crop((left, top, right, bottom))
    except Exception as e:
        raise ValueError(f"Failed to crop the image: {e}")


def _read_pdf_as_imgs(pdf_document: fitz.Document) -> List[Image.Image]:
    images = []
    try:
        if not pdf_document.is_pdf:
            raise ValueError("The document is not in PDF format.")
        if pdf_document.needs_pass:
            raise ValueError("The PDF document is password protected.")
        page_numbers = list(range(pdf_document.page_count))

        for n in page_numbers:
            page = pdf_document[n]
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(image)

    except ValueError as e:
        logging.error(f"ValueError: {e}")
    except IndexError as e:
        logging.error(f"Page index out of range: {e}")
    except Exception as e:
        logging.error(f"An error occurred while reading the PDF: {e}")

    return images
