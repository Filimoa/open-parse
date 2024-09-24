import base64
import io
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock

from pdfminer.layout import LTAnno, LTChar
from PIL import Image, ImageChops

from openparse.pdf import Pdf
from openparse.schemas import NodeVariant, TextSpan
from openparse.text.pdfminer.core import (
    CharElement,
    _extract_chars,
    _group_chars_into_spans,
    ingest,
)

raw_chars = [
    CharElement(text="1", fontname="bold", size=9.0),
    CharElement(text=".", fontname="bold", size=9.0),
    CharElement(text=" ", fontname="bold", size=9.0),
    CharElement(text="P", fontname="bold", size=9.0),
    CharElement(text="A", fontname="bold", size=9.0),
    CharElement(text="R", fontname="bold", size=9.0),
    CharElement(text="T", fontname="bold", size=9.0),
    CharElement(text="I", fontname="bold", size=9.0),
    CharElement(text="E", fontname="bold", size=9.0),
    CharElement(text="S", fontname="bold", size=9.0),
    CharElement(text=":", fontname="bold", size=9.0),
    CharElement(text=" ", fontname="", size=9.0),
    CharElement(text=" ", fontname="", size=9.0),
]
spans = [
    TextSpan(text="1. PARTIES: ", is_bold=True, is_italic=False, size=9.0),
]


def test_group_chars_into_spans():
    # Test the basic functionality with the given raw_chars and spans
    result = _group_chars_into_spans(raw_chars)
    assert len(result) == len(
        spans
    ), "The number of spans should match the expected count."

    for result_span, expected_span in zip(result, spans):
        assert (
            result_span.text == expected_span.text
        ), f"Expected text '{expected_span.text}', got '{result_span.text}'"
        assert (
            result_span.is_bold == expected_span.is_bold
        ), f"Expected is_bold {expected_span.is_bold}, got {result_span.is_bold}"
        assert (
            result_span.is_italic == expected_span.is_italic
        ), f"Expected is_italic {expected_span.is_italic}, got {result_span.is_italic}"
        assert (
            result_span.size == expected_span.size
        ), f"Expected size {expected_span.size}, got {result_span.size}"

    # Test with mixed styles to ensure correct grouping
    mixed_chars = [
        CharElement(text="H", fontname="bold", size=9.0),
        CharElement(text="e", fontname="italic", size=9.0),
        CharElement(text="l", fontname="bold", size=9.0),
        CharElement(text="l", fontname="bold", size=9.0),
        CharElement(text="o", fontname="", size=9.0),
        CharElement(text=" ", fontname="", size=9.0),
        CharElement(text="W", fontname="boldItalic", size=9.0),
        CharElement(text="o", fontname="boldItalic", size=9.0),
        CharElement(text="r", fontname="boldItalic", size=9.0),
        CharElement(text="l", fontname="boldItalic", size=9.0),
        CharElement(text="d", fontname="boldItalic", size=9.0),
    ]
    mixed_spans = [
        TextSpan(text="H", is_bold=True, is_italic=False, size=9.0),
        TextSpan(text="e", is_bold=False, is_italic=True, size=9.0),
        TextSpan(text="ll", is_bold=True, is_italic=False, size=9.0),
        TextSpan(text="o ", is_bold=False, is_italic=False, size=9.0),
        TextSpan(text="World", is_bold=True, is_italic=True, size=9.0),
    ]
    mixed_result = _group_chars_into_spans(mixed_chars)
    assert len(mixed_result) == len(
        mixed_spans
    ), "The number of spans in mixed styles should match the expected count."

    for result_span, expected_span in zip(mixed_result, mixed_spans):
        assert (
            result_span.text == expected_span.text
        ), f"Expected text '{expected_span.text}', got '{result_span.text}' in mixed styles"
        assert (
            result_span.is_bold == expected_span.is_bold
        ), f"Expected is_bold {expected_span.is_bold}, got {result_span.is_bold} in mixed styles"
        assert (
            result_span.is_italic == expected_span.is_italic
        ), f"Expected is_italic {expected_span.is_italic}, got {result_span.is_italic} in mixed styles"
        assert (
            result_span.size == expected_span.size
        ), f"Expected size {expected_span.size}, got {result_span.size} in mixed styles"

    # Add more tests here for additional scenarios like empty inputs, inputs with only spaces, etc.


def _char_data_to_text_line(char_data: List[Tuple[str, str, float]]):
    text_line = []
    for text, fontname, size in char_data:
        # LTAnno does not have fontname and size attributes
        if fontname is None and size is None:
            anno = MagicMock(spec=LTAnno)
            anno.get_text.return_value = text
            text_line.append(anno)
        else:
            char = MagicMock(spec=LTChar)
            char.get_text.return_value = text
            char.fontname = fontname
            char.size = size
            text_line.append(char)
    return text_line


def test_extract_chars():
    char_data = [
        ("A", "Arial-Bold", 12.0),
        ("b", "Arial-Bold", 12.0),
        ("o", "Arial-Bold", 12.0),
        ("u", "Arial-Bold", 12.0),
        ("t", "Arial-Bold", 12.0),
        (" ", "Arial-Bold", 12.0),
        ("w", "Arial-Bold", 12.0),
        ("h", "Arial-Bold", 12.0),
        ("o", "Arial-Bold", 12.0),
    ]

    expected_output = [
        CharElement(text="A", fontname="Arial-Bold", size=12.0),
        CharElement(text="b", fontname="Arial-Bold", size=12.0),
        CharElement(text="o", fontname="Arial-Bold", size=12.0),
        CharElement(text="u", fontname="Arial-Bold", size=12.0),
        CharElement(text="t", fontname="Arial-Bold", size=12.0),
        CharElement(text=" ", fontname="Arial-Bold", size=12.0),
        CharElement(text="w", fontname="Arial-Bold", size=12.0),
        CharElement(text="h", fontname="Arial-Bold", size=12.0),
        CharElement(text="o", fontname="Arial-Bold", size=12.0),
    ]

    text_line = _char_data_to_text_line(char_data)
    result = _extract_chars(text_line)
    assert len(result) == 9

    # Assert the result matches the expected output
    assert result == expected_output


def test_extract_chars_with_ltannos():
    # Data for LTChar and LTAnno mock instances interleaved
    char_data = [
        ("A", "Arial-Bold", 12.0),
        ("b", "Arial-Bold", 12.0),
        ("o", "Arial-Bold", 12.0),
        ("u", "Arial-Bold", 12.0),
        ("t", "Arial-Bold", 12.0),
        (" ", None, None),  # LTAnno
        ("w", "Arial-Bold", 12.0),
        ("h", "Arial-Bold", 12.0),
        ("o", "Arial-Bold", 12.0),
        (" ", None, None),  # LTAnno
        (" ", None, None),  # LTAnno
        ("A", "Arial-Bold", 12.0),
        ("u", "Arial-Bold", 12.0),
        ("t", "Arial-Bold", 12.0),
        ("h", "Arial-Bold", 12.0),
        ("o", "Arial-Bold", 12.0),
        ("r", "Arial-Bold", 12.0),
        ("\n", None, None),  # LTAnno
    ]

    text_line = _char_data_to_text_line(char_data)

    # Expected output
    expected_output = [
        CharElement(text="A", fontname="Arial-Bold", size=12.0),
        CharElement(text="b", fontname="Arial-Bold", size=12.0),
        CharElement(text="o", fontname="Arial-Bold", size=12.0),
        CharElement(text="u", fontname="Arial-Bold", size=12.0),
        CharElement(text="t", fontname="Arial-Bold", size=12.0),
        CharElement(text=" ", fontname="Arial-Bold", size=12.0),
        CharElement(text="w", fontname="Arial-Bold", size=12.0),
        CharElement(text="h", fontname="Arial-Bold", size=12.0),
        CharElement(text="o", fontname="Arial-Bold", size=12.0),
        CharElement(text=" ", fontname="Arial-Bold", size=12.0),
        CharElement(text=" ", fontname="Arial-Bold", size=12.0),
        CharElement(text="A", fontname="Arial-Bold", size=12.0),
        CharElement(text="u", fontname="Arial-Bold", size=12.0),
        CharElement(text="t", fontname="Arial-Bold", size=12.0),
        CharElement(text="h", fontname="Arial-Bold", size=12.0),
        CharElement(text="o", fontname="Arial-Bold", size=12.0),
        CharElement(text="r", fontname="Arial-Bold", size=12.0),
        CharElement(text="\n", fontname="Arial-Bold", size=12.0),
    ]

    # Call _extract_chars
    result = _extract_chars(text_line)
    assert len(result) == 18

    # Assert the result matches the expected output
    assert result == expected_output


def _images_are_similar(img1_bytes, img2_bytes, max_pct_diff=1.0, pixel_threshold=10):
    """
    Compare two images and determine if the percentage of differing pixels is below a threshold.

    :param img1_bytes: Byte content of the first image.
    :param img2_bytes: Byte content of the second image.
    :param max_pct_diff: Maximum allowed percentage of differing pixels.
    :param pixel_threshold: Per-pixel difference threshold to consider a pixel as different.
    :return: Boolean indicating if images are similar within the allowed percentage difference.
    """
    img1 = Image.open(io.BytesIO(img1_bytes)).convert("RGB")
    img2 = Image.open(io.BytesIO(img2_bytes)).convert("RGB")

    if img1.size != img2.size:
        print(f"Image sizes do not match: {img1.size} vs {img2.size}")
        return False

    diff = ImageChops.difference(img1, img2)

    diff_gray = diff.convert("L")

    differing_pixels = sum(
        1 for pixel in diff_gray.getdata() if pixel > pixel_threshold
    )
    total_pixels = img1.size[0] * img1.size[1]
    pct_diff = (differing_pixels / total_pixels) * 100

    print(f"Percentage of differing pixels: {pct_diff:.2f}%")
    return pct_diff <= max_pct_diff


def test_parse_pdf_with_images():
    doc_with_image_path = Path("src/tests/sample_data/pdf-with-image.pdf")
    pdf = Pdf(doc_with_image_path)

    elems = ingest(pdf)
    assert elems[-1].variant == NodeVariant.IMAGE
    assert elems[-1].image_mimetype == "image/jpeg"
    extracted_image_data = base64.b64decode(elems[-1].image)

    # Read the raw image data
    raw_image_path = Path("src/tests/sample_data/europe.jpg")
    with raw_image_path.open("rb") as img_file:
        raw_image_data = img_file.read()

    assert _images_are_similar(raw_image_data, extracted_image_data)
