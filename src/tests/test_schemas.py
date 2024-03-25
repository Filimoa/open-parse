from src.consts import MAX_EMBEDDING_TOKENS
from src.schemas import LineElement, TextElement, Bbox, Node
import pytest


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
    element1 = LineElement(text="Element 1", bbox=bbox1)
    element2 = LineElement(text="Element 2", bbox=bbox2)
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


@pytest.mark.parametrize(
    "criteria, expected_combine",
    [
        ("both_small", True),
        ("either_stub", True),
    ],
)
def test_node_combination_spatially(create_text_element, criteria, expected_combine):
    node1 = create_text_element(1, 0, 0, 2, 2)
    node2 = create_text_element(1, 1.5, 1.5, 3.5, 3.5)

    combined_nodes = _combine_nodes_spatially(
        [node1, node2], x_error_margin=0, y_error_margin=0, criteria=criteria
    )

    if expected_combine:
        assert len(combined_nodes) == 1, "Nodes should have been combined"
    else:
        assert len(combined_nodes) == 2, "Nodes should not have been combined"


def test_nodes_on_different_pages_should_not_combine(create_text_element):
    node1 = create_text_element(1, 0, 0, 1, 1)
    node2 = create_text_element(2, 0, 0, 1, 1)

    combined_nodes = _combine_nodes_spatially(
        [node1, node2], x_error_margin=0, y_error_margin=0, criteria="both_small"
    )

    assert len(combined_nodes) == 2, "Nodes on different pages should not combine"
