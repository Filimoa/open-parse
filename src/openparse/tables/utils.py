import logging
from typing import List, Literal, Tuple

from PIL import Image  # type: ignore

Size = Tuple[int, int]
BBox = Tuple[float, float, float, float]


###################
### IMAGE UTILS ###
###################


def crop_img_with_padding(
    image: Image.Image, bbox: BBox, padding_pct: float
) -> Image.Image:
    """
    Adds whitespace outside the image. Recomennded by the model authors.
    """
    if padding_pct < 0:
        raise ValueError("Padding percentage must be non-negative")
    if padding_pct >= 1:
        raise ValueError("Padding percentage must be less than 1")

    left, top, right, bottom = map(int, bbox)

    if not (0 <= left < right <= image.width) or not (
        0 <= top < bottom <= image.height
    ):
        raise ValueError("Bounding box is out of the image boundaries")

    try:
        cropped_image = image.crop((left, top, right, bottom))

        width = right - left
        height = bottom - top
        padding_x = int(width * padding_pct)
        padding_y = int(height * padding_pct)

        new_width = width + 2 * padding_x
        new_height = height + 2 * padding_y

        padded_image = Image.new("RGB", (new_width, new_height), color="white")
        padded_image.paste(cropped_image, (padding_x, padding_y))

        return padded_image

    except Exception as e:
        raise ValueError(f"Failed to crop the image: {e}")


def doc_to_imgs(doc) -> List[Image.Image]:
    images = []
    try:
        if not doc.is_pdf:
            raise ValueError("The document is not in PDF format.")
        if doc.needs_pass:
            raise ValueError("The PDF document is password protected.")
        page_numbers = list(range(doc.page_count))

        for n in page_numbers:
            page = doc[n]
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


def display_cells_on_img(
    image: Image.Image,
    cells,
    show_cell_types: Literal["all", "headers", "rows", "columns"] = "all",
    use_blank_image: bool = False,
    min_cell_confidence: float = 0.95,
) -> None:
    """
    Used for debugging to visualize the detected cells on the cropped table image.
    """
    try:
        from IPython.display import display  # type: ignore
        from PIL import ImageDraw  # type: ignore
    except ImportError:
        logging.error(
            "IPython or PIL is not installed to display cells on the image. Skipping"
        )
        return

    cropped_table_visualized = image.copy()
    if use_blank_image:
        cropped_table_visualized = Image.new("RGB", image.size, color="white")
    draw = ImageDraw.Draw(cropped_table_visualized)

    for cell in cells:
        if cell.confidence < min_cell_confidence:
            continue

        if show_cell_types == "headers" and not cell.is_header:
            continue
        elif show_cell_types == "rows" and not cell.is_row:
            continue
        elif show_cell_types == "columns" and not cell.is_column:
            continue

        draw.rectangle(cell.bbox, outline="red")

    display(cropped_table_visualized)
