import pytest
from openparse.schemas import (
    Bbox,
    LineElement,
    Node,
    TextElement,
    TextSpan,
    bullet_regex,
)
from openparse import consts

BOLD_FLAG = 2**4
ITALIC_FLAG = 2**1
MONOSPACED_FLAG = 0


@pytest.fixture
def create_text_element():
    def _create_text_element(page, x0, y0, x1, y1, text="Sample text"):
        element = TextElement(
            text=text,
            lines=[],
            bbox=Bbox(
                page=page, page_height=11, page_width=8.5, x0=x0, y0=y0, x1=x1, y1=y1
            ),
            position=0,
        )
        return Node(elements=[element])

    return _create_text_element


def create_node(elements):
    return Node(elements=elements)


@pytest.fixture
def bold_span():
    return TextSpan(text="Bold Text", is_bold=True, is_italic=False, size=12)


@pytest.fixture
def italic_span():
    return TextSpan(text="Italic Text", is_bold=False, is_italic=True, size=12)


@pytest.fixture
def regular_span():
    return TextSpan(text="Regular Text", is_bold=False, is_italic=False, size=12)


def mixed_span():
    return TextSpan(text="Bold and Italic", is_bold=True, is_italic=True, size=12)


########################
### SPAN + LINE TESTS ###
########################


def test_formatted_text_no_adjacent(bold_span, italic_span, regular_span):
    # You'll need to implement or adjust the formatted_text method to use is_bold and is_italic
    assert bold_span.formatted_text() == "**Bold Text**", "Bold formatting failed"
    assert italic_span.formatted_text() == "*Italic Text*", "Italic formatting failed"
    assert (
        regular_span.formatted_text() == "Regular Text"
    ), "Regular text altered incorrectly"


# Update or remove test cases related to flags, since your logic now directly uses the boolean fields


def test_formatted_text_with_same_style_adjacent(bold_span):
    next_span_same_style = TextSpan(
        text=" Next", is_bold=True, is_italic=False, size=12
    )

    formatted_text = bold_span.formatted_text(next_span=next_span_same_style)
    assert formatted_text.startswith("**"), formatted_text
    assert not formatted_text.endswith("**"), formatted_text


def test_formatted_text_with_different_style_adjacent(bold_span, regular_span):
    formatted_text = bold_span.formatted_text(next_span=regular_span)
    assert formatted_text.endswith("**"), formatted_text


def test_formatted_text_edge_cases():
    empty_span = TextSpan(text="", is_bold=False, is_italic=False, size=12)

    assert empty_span.formatted_text() == "", "Empty span text formatting altered"

    no_style_span = TextSpan(text="No Style", is_bold=False, is_italic=False, size=12)
    assert (
        no_style_span.formatted_text() == "No Style"
    ), "No style span text formatting altered"


def test_mixed_bold_and_italic_within_same_span():
    mixed_span = TextSpan(text="Bold and Italic", is_bold=True, is_italic=True, size=12)
    assert (
        mixed_span.formatted_text() == "***Bold and Italic***"
    ), "Mixed bold and italic formatting failed"


def test_adjacent_spans_with_opposite_styles(bold_span, italic_span):
    formatted_text = bold_span.formatted_text(next_span=italic_span)
    assert formatted_text.endswith("**"), "Bold span did not end with bold markdown"
    # Assuming the italic_span formats itself considering the adjacent bold_span
    formatted_text_next = italic_span.formatted_text(previous_span=bold_span)
    assert formatted_text_next.startswith(
        "*"
    ), "Italic span did not start with italic markdown after bold"


def test_line_with_mixed_style_spans():
    # punting for now, need to add white space support so we don't get '***' instead of '** *'
    # spans = [
    #     TextSpan(text="Regular ", flags=0, size=12),
    #     TextSpan(text="Bold", flags=BOLD_FLAG, size=12),
    #     TextSpan(text=" Italic", flags=ITALIC_FLAG, size=12),
    # ]
    # line_element = LineElement(bbox=(0, 0, 0, 0), spans=spans)
    # assert (
    #     line_element.text == "Regular **Bold** *Italic*"
    # ), "Line with mixed styles formatted incorrectly"
    ...


def test_various_spans_found_in_lease_agreement():
    # Test Case 1: Mixed bold and regular text
    spans_mixed_bold = [
        TextSpan(text="T", is_bold=True, is_italic=False, size=14.0),
        TextSpan(text="ENNYSON ", is_bold=True, is_italic=False, size=11.0),
        TextSpan(text="P", is_bold=True, is_italic=False, size=14.0),
        TextSpan(text="LACE ", is_bold=True, is_italic=False, size=11.0),
        TextSpan(text="L", is_bold=True, is_italic=False, size=14.0),
        TextSpan(text="EASE ", is_bold=True, is_italic=False, size=11.0),
        TextSpan(text="A", is_bold=True, is_italic=False, size=14.0),
        TextSpan(text="GREEMENT", is_bold=True, is_italic=False, size=11.0),
        TextSpan(text=" ", is_bold=True, is_italic=False, size=25.0),
    ]
    line_element_mixed_bold = LineElement(bbox=(0, 0, 0, 0), spans=spans_mixed_bold)
    assert (
        line_element_mixed_bold.text == "**TENNYSON PLACE LEASE AGREEMENT**"
    ), "Failed mixed bold styling"

    # Test Case 2: Starting with bold and ending with regular text
    spans_start_bold_end_regular = [
        TextSpan(text="1.", is_bold=True, is_italic=False, size=9.0),
        TextSpan(text=" ", is_bold=False, is_italic=False, size=9.0),
        TextSpan(text="PARTIES:", is_bold=True, is_italic=False, size=9.0),
        TextSpan(text="  ", is_bold=False, is_italic=False, size=9.0),
    ]
    line_element_start_bold_end_regular = LineElement(
        bbox=(0, 0, 0, 0), spans=spans_start_bold_end_regular
    )
    assert (
        line_element_start_bold_end_regular.text == "**1.** **PARTIES:**"
    ), "Failed handling start bold end regular"

    # Test Case 3: Complex case with bold and regular spans
    spans_complex = [
        TextSpan(
            text="THIS RENTAL LEASE AGREEMENT (hereinafter “Lease” or “Agreement”) dated ",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(text="1/12/2003 12:36:16 PM", is_bold=True, is_italic=False, size=9.0),
        TextSpan(
            text=" between Hacker Apartment ", is_bold=False, is_italic=False, size=9.0
        ),
        TextSpan(
            text='Services, Inc. as Owner or as agent for the Owner (hereinafter "Agent") ',
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(text="Lebron James", is_bold=True, is_italic=False, size=9.0),
        TextSpan(
            text=' (collectively hereinafter "Resident").  ',
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="Resident along with the following persons, shall be authorized occupants.",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
    ]

    line_element_complex = LineElement(bbox=(0, 0, 0, 0), spans=spans_complex)
    expected_complex = (
        "THIS RENTAL LEASE AGREEMENT (hereinafter “Lease” or “Agreement”) dated**1/12/2003 12:36:16 PM**"
        'between Hacker Apartment Services, Inc. as Owner or as agent for the Owner (hereinafter "Agent")'
        '**Lebron James**(collectively hereinafter "Resident").  '
        "Resident along with the following persons, shall be authorized occupants."
    )

    assert (
        line_element_complex.text == expected_complex.strip()
    ), "Failed complex case handling"

    # Test Case 4: Big paragraph with mixed styles
    spans_legal = [
        TextSpan(
            text="In any disputed court action where the court resolves the dispute and determines the prevailing party, the court shall also award to the ",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="prevailing party its attorneys’ fees and costs and the non-prevailing party shall be liable to the prevailing party for payment of any court ",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="awarded attorneys’ fees and costs. Resident agrees to pay eighteen percent (18%) interest compounded annually on all unpaid rent, amounts, ",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="or damages owed by Resident, except for late fees, from that date of Landlord’s final accounting until such time Resident pays all outstanding ",
            is_bold=False,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(text="amounts.  ", is_bold=False, is_italic=False, size=9.0),
        TextSpan(
            text="Agent and Resident agree that any action or proceeding arising out of or in any way connected with this Agreement, ",
            is_bold=True,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="regardless of whether such claim is based on contract, tort, or other legal theory, shall be heard by a court sitting without a jury and ",
            is_bold=True,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(
            text="thus Resident hereby waives all rights to a trial by jury",
            is_bold=True,
            is_italic=False,
            size=9.0,
        ),
        TextSpan(text=". ", is_bold=True, is_italic=False, size=9.0),
    ]

    expected_legal_text = (
        "In any disputed court action where the court resolves the dispute and determines the prevailing party, the court shall also award to the "
        "prevailing party its attorneys’ fees and costs and the non-prevailing party shall be liable to the prevailing party for payment of any court "
        "awarded attorneys’ fees and costs. Resident agrees to pay eighteen percent (18%) interest compounded annually on all unpaid rent, amounts, "
        "or damages owed by Resident, except for late fees, from that date of Landlord’s final accounting until such time Resident pays all outstanding "
        "amounts.**Agent and Resident agree that any action or proceeding arising out of or in any way connected with this Agreement, "
        "regardless of whether such claim is based on contract, tort, or other legal theory, shall be heard by a court sitting without a jury and "
        "thus Resident hereby waives all rights to a trial by jury.**"
    )

    line_element = LineElement(bbox=(0, 0, 0, 0), spans=spans_legal)
    assert (
        line_element.text.strip() == expected_legal_text.strip()
    ), "Failed handling legal section with mixed styles"


############################
### ELEMENT + NODE TESTS ###
############################


@pytest.mark.parametrize(
    "bbox1, bbox2, error_margin, expected_result",
    [
        # Non-overlapping boxes, no error margin
        ((0, 0, 1, 1), (2, 2, 3, 3), 0, False),
        # Overlapping boxes, no error margin
        ((0, 0, 2, 2), (1, 1, 3, 3), 0, True),
        # Adjacent boxes touching at edges, considered as overlapping because of identical edges
        ((0, 0, 1, 1), (1, 1, 2, 2), 0, True),
        # Boxes near each other but not touching, with error margin, turning non-overlapping to overlapping
        ((0, 0, 1, 1), (1.1, 1.1, 2.1, 2.1), 0.15, True),
    ],
)
def test_line_element_overlaps(bbox1, bbox2, error_margin, expected_result):
    element1 = LineElement(text="Element 1", bbox=bbox1, spans=[])
    element2 = LineElement(text="Element 2", bbox=bbox2, spans=[])
    assert element1.overlaps(element2, error_margin=error_margin) == expected_result


@pytest.mark.parametrize(
    "bbox1, bbox2, page1, page2, error_margin, expected_result",
    [
        # Non-overlapping boxes on the same page
        ((0, 0, 1, 1), (2, 2, 3, 3), 1, 1, 0, False),
        # Overlapping boxes on the same page
        ((0, 0, 2, 2), (1, 1, 3, 3), 1, 1, 0, True),
        # Adjacent boxes on the same page, considered as overlapping
        ((0, 0, 1, 1), (1, 1, 2, 2), 1, 1, 0, True),
        # Boxes near but not touching on the same page, error margin makes them overlap
        ((0, 0, 1, 1), (1.1, 1.1, 2.1, 2.1), 1, 1, 0.15, True),
        # Boxes on different pages, should not overlap regardless of their positions
        ((0, 0, 1, 1), (0, 0, 1, 1), 1, 2, 0, False),
    ],
)
def test_text_element_overlaps_different_pages(
    bbox1, bbox2, page1, page2, error_margin, expected_result
):
    element1_bbox = Bbox(
        page=page1,
        page_height=11,
        page_width=8.5,
        x0=bbox1[0],
        y0=bbox1[1],
        x1=bbox1[2],
        y1=bbox1[3],
        lines=[],
    )
    element2_bbox = Bbox(
        page=page2,
        page_height=11,
        page_width=8.5,
        x0=bbox2[0],
        y0=bbox2[1],
        x1=bbox2[2],
        y1=bbox2[3],
        lines=[],
    )
    element1 = TextElement(text="Element 1", lines=[], bbox=element1_bbox, position=0)
    element2 = TextElement(text="Element 2", lines=[], bbox=element2_bbox, position=1)

    assert (
        element1.overlaps(
            element2, x_error_margin=error_margin, y_error_margin=error_margin
        )
        == expected_result
    )


def test_node_bbox():
    sample_elements = [
        TextElement(
            text="Element on page 1",
            lines=[],
            bbox=Bbox(
                page=1, page_height=800, page_width=600, x0=50, y0=100, x1=150, y1=200
            ),
        ),
        TextElement(
            text="Another element on page 1",
            lines=[],
            bbox=Bbox(
                page=1, page_height=800, page_width=600, x0=100, y0=150, x1=200, y1=250
            ),
        ),
        TextElement(
            text="Element on page 2",
            lines=[],
            bbox=Bbox(
                page=2, page_height=800, page_width=600, x0=60, y0=110, x1=160, y1=210
            ),
        ),
    ]

    node = Node(elements=sample_elements)

    bboxes = node.bbox

    # Assertions
    assert len(bboxes) == 2, "There should be two bounding boxes for two pages."

    # Assertions for page 1 bounding box
    bbox_page_1 = [bbox for bbox in bboxes if bbox.page == 1][0]
    assert bbox_page_1.x0 == 50, "Incorrect x0 for page 1"
    assert bbox_page_1.y0 == 100, "Incorrect y0 for page 1"
    assert bbox_page_1.x1 == 200, "Incorrect x1 for page 1"
    assert bbox_page_1.y1 == 250, "Incorrect y1 for page 1"

    # Assertions for page 2 bounding box
    bbox_page_2 = [bbox for bbox in bboxes if bbox.page == 2][0]
    assert bbox_page_2.x0 == 60, "Incorrect x0 for page 2"
    assert bbox_page_2.y0 == 110, "Incorrect y0 for page 2"
    assert bbox_page_2.x1 == 160, "Incorrect x1 for page 2"
    assert bbox_page_2.y1 == 210, "Incorrect y1 for page 2"


@pytest.mark.parametrize(
    "bbox1, bbox2, page1, page2, x_error_margin, y_error_margin, expected",
    [
        # Existing test cases simplified with page parameters
        (
            (0, 0, 1, 1),
            (2, 2, 3, 3),
            1,
            1,
            0,
            0,
            False,
        ),  # Non-overlapping on the same page
        ((0, 0, 2, 2), (1, 1, 3, 3), 1, 1, 0, 0, True),  # Overlapping on the same page
        ((0, 0, 1, 1), (1, 1, 2, 2), 1, 1, 0, 0, True),  # Adjacent on the same page
        ((0, 0, 1, 1), (0, 0, 1, 1), 1, 2, 0, 0, False),  # Same bbox, different pages
        # Additional test cases
        (
            (0, 0, 1, 1),
            (1.05, 1.05, 2.05, 2.05),
            1,
            1,
            0.1,
            0.1,
            True,
        ),  # Near but not touching, error margin causes overlap
        #    ((0, 0, 0, 0), (0, 0, 1, 1), 1, 1, 0, 0, False),  # Node with zero area SHOULD NEVER HAPPEN
        (
            (0, 0, 3, 3),
            (1, 1, 2, 2),
            1,
            1,
            0,
            0,
            True,
        ),  # One node completely encompasses another
    ],
)
def test_node_overlaps(
    create_text_element,
    bbox1,
    bbox2,
    page1,
    page2,
    x_error_margin,
    y_error_margin,
    expected,
):
    node1 = create_text_element(
        page=page1, x0=bbox1[0], y0=bbox1[1], x1=bbox1[2], y1=bbox1[3]
    )
    node2 = create_text_element(
        page=page2, x0=bbox2[0], y0=bbox2[1], x1=bbox2[2], y1=bbox2[3]
    )

    assert (
        node1.overlaps(
            node2, x_error_margin=x_error_margin, y_error_margin=y_error_margin
        )
        == expected
    )


##### BULLETS


@pytest.mark.parametrize(
    "case",
    [
        "- This is a dash bullet point",
        "• This is a bullet point",
        "* This is an asterisk bullet point",
        "1. This is a numbered bullet point",
        "(1) This is a parenthesized number bullet point",
        "(a) This is a parenthesized letter bullet point",
        "A. This is a lettered bullet point",
    ],
)
def test_bullet_point_detection_match(case):
    assert bullet_regex.match(case) is not None, f"Should match: {case}"


# Test cases that should NOT match the pattern
@pytest.mark.parametrize(
    "case",
    [
        "This is a regular line",
        ".1 This is not a numbered bullet point",
        "( 1) This format is not covered",
        "A- This is not a bullet point",
        "**RULE 10 - CLASSIFICATION**",
    ],
)
def test_bullet_point_detection_no_match(case):
    assert bullet_regex.match(case) is None, f"Should not match: {case}"


def test_text_element_with_bullets():
    # This test is to ensure that the text element with bullets is correctly processed
    # The text element has a single line with multiple bullet points
    # The bullet points are expected to be separated into individual lines
    # The bullet points are expected to be separated by a newline
    # first
    text_elem_without_bullets = TextElement(
        text=(
            "regulatory complexities. For example, several of our products are not generally available in China. "
            "We also outsource certain operational functions to third parties globally. If we fail to deploy, manage, "
            "or oversee our international operations successfully, our business may suffer. In addition, we are subject "
            "to a variety of risks inherent in doing business internationally, including:"
        ),
        lines=(),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=659.1,
            x1=594.0,
            y1=711.3,
        ),
    )
    node = Node(elements=[text_elem_without_bullets])
    assert node.starts_with_bullet == False
    assert node.ends_with_bullet == False

    # second
    text_elem_starts_with_bullet = TextElement(
        text=(
            f"- This is the first bullet point. {consts.ELEMENT_DELIMETER}"
            "Additional information follows the bullet point without a new bullet."
        ),
        lines=(),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=659.1,
            x1=594.0,
            y1=711.3,
        ),
    )
    node = Node(elements=[text_elem_starts_with_bullet])
    assert node.starts_with_bullet == True
    assert node.ends_with_bullet == False

    # third
    text_elem_ends_with_bullet = TextElement(
        text=(
            f"This is some introductory text. {consts.ELEMENT_DELIMETER}"
            "- And this is a bullet point that concludes the text element."
        ),
        lines=(),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=659.1,
            x1=594.0,
            y1=711.3,
        ),
    )
    node = Node(elements=[text_elem_ends_with_bullet])
    assert node.starts_with_bullet == False
    assert node.ends_with_bullet == True

    # fourth
    text_elem_contains_bullets = TextElement(
        text=(
            f"- First bullet point starts the element. {consts.ELEMENT_DELIMETER}"
            f"- Second bullet point follows.  {consts.ELEMENT_DELIMETER}"
            f"Some intermediate text that doesn't start with a bullet. {consts.ELEMENT_DELIMETER}"
            "- Third bullet point ends the element."
        ),
        lines=(),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=659.1,
            x1=594.0,
            y1=711.3,
        ),
    )
    node = Node(elements=[text_elem_contains_bullets])
    assert node.starts_with_bullet == True
    assert node.ends_with_bullet == True
