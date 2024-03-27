from openparse.schemas import TextSpan
from openparse.text.pdfminer.core import CharElement, _group_chars_into_spans

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
