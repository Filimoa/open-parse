import pytest
from openparse.processing import (
    ProcessingStep,
    RemoveTextInsideTables,
    RemoveRepeatedElements,
    RemoveFullPageStubs,
    CombineNodesSpatially,
    CombineBullets,
    CombineHeadingsWithClosestText,
    RemoveNodesBelowNTokens,
)
from openparse import consts
from openparse.schemas import (
    Node,
    TextElement,
    TableElement,
    Bbox,
    LineElement,
    TextSpan,
)
from unittest.mock import MagicMock, PropertyMock


class TestNode:
    __test__ = False

    def __init__(self, node, is_stub=False, num_pages=1, is_small=True):
        self.node = node
        self.is_stub = is_stub
        self.num_pages = num_pages
        self.is_small = is_small
        # Mimic other Node properties and methods as necessary for the tests

    def __getattr__(self, item):
        return getattr(self.node, item)


def create_text_node(text, x0, y0, x1, y1, page=1):
    return Node(
        elements=(
            TextElement(
                text=text,
                lines=(),
                bbox=Bbox(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page=page,
                    page_height=5000,
                    page_width=5000,
                ),
            ),
        ),
    )


def create_table_node(x0, y0, x1, y1, page=1):
    return Node(
        elements=(
            TableElement(
                text="",
                bbox=Bbox(
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    page=page,
                    page_height=5000,
                    page_width=5000,
                ),
            ),
        )
    )


### RemoveTextInsideTables tests ###


def test_no_tables_in_document():
    nodes = [create_text_node("Sample text", 0, 0, 10, 10)]
    expected = nodes  # No change expected
    assert RemoveTextInsideTables().process(nodes) == expected


def test_text_outside_tables():
    nodes = [
        create_table_node(50, 50, 100, 100),
        create_text_node("Outside text", 10, 10, 20, 20),
    ]
    expected = nodes  # Expect no changes to the text node
    assert RemoveTextInsideTables().process(nodes) == expected


def test_text_inside_tables():
    nodes = [
        create_table_node(50, 50, 100, 100),
        create_text_node("Inside text", 60, 60, 70, 70),
    ]
    expected = [nodes[0]]  # Expect only the table node to remain
    assert RemoveTextInsideTables().process(nodes) == expected


def test_text_on_table_border():
    nodes = [
        create_table_node(50, 50, 100, 100),
        create_text_node("Border text", 50, 50, 100, 100),
    ]
    expected = [nodes[0]]  # Assuming border text is considered inside
    assert RemoveTextInsideTables().process(nodes) == expected


def test_overlapping_tables():
    nodes = [
        create_table_node(40, 40, 80, 80),
        create_table_node(60, 60, 100, 100),
        create_text_node("Overlapping text", 70, 70, 75, 75),
    ]
    expected = nodes[:2]  # Expect the text to be removed
    assert RemoveTextInsideTables().process(nodes) == expected


def test_empty_tables():
    nodes = [
        create_table_node(50, 50, 100, 100),
    ]
    expected = nodes  # No change expected
    assert RemoveTextInsideTables().process(nodes) == expected


def test_multi_page_documents():
    nodes = [
        create_table_node(50, 50, 100, 100, page=1),
        create_text_node("Page 1 text", 60, 60, 70, 70, page=1),
        create_text_node("Page 2 text", 60, 60, 70, 70, page=2),
    ]
    expected = [
        nodes[0],
        nodes[2],
    ]  # Expect the text on page 1 to be removed, page 2 text remains
    assert RemoveTextInsideTables().process(nodes) == expected


### RemoveTextInsideTables tests ###


def test_page_exceeds_max_area_percentage():
    # Content area significantly larger than 1% of the page; should be retained
    node = create_text_node("Large content", 0, 0, 2500, 2500)
    test_node = TestNode(node)
    assert RemoveFullPageStubs(max_area_pct=0.01).process([test_node]) == [
        test_node
    ], "Node should be retained"


def test_page_above_max_area_percentage():
    # Large content area; should be considered excessive and dropped
    node = create_text_node("small heading", 0, 0, 3500, 3500)
    assert (
        RemoveFullPageStubs(max_area_pct=0.01).process([node]) == []
    ), "Node should be dropped as it exceeds the max area percentage"


def test_multi_page_nodes():
    # Node spanning multiple pages; should be retained regardless of content area
    node = create_text_node("Multi-page content", 0, 0, 100, 100)
    test_node = TestNode(node, num_pages=2)
    assert RemoveFullPageStubs(max_area_pct=0.01).process([test_node]) == [
        test_node
    ], "Multi-page nodes should always be retained"


def test_non_stub_nodes():
    # Node marked as non-stub; should be retained regardless of content area
    node = create_text_node("Non-stub content", 0, 0, 100, 100)
    test_node = TestNode(node, is_stub=False)
    assert RemoveFullPageStubs(max_area_pct=0.01).process([test_node]) == [
        test_node
    ], "Non-stub nodes should always be retained"


### RemoveRepeatedElements tests ###


def test_remove_repeated_elements():
    sample_bbox = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}

    nodes = [
        create_text_node(text="Hello, world!", **sample_bbox),
        create_text_node(text="Hello, world!", **sample_bbox),
        create_text_node(text="Unique text", **sample_bbox),
        create_text_node(
            text="Hello, world!", **sample_bbox
        ),  # This text repeats 3 times.
        create_text_node(text="Another unique text", **sample_bbox),
    ]

    processor = RemoveRepeatedElements(threshold=2)
    processed_nodes = processor.process(nodes)

    # "Hello, world!" is removed completely because it appears more than twice
    expected_texts = {"Unique text", "Another unique text"}
    processed_texts = {node.text for node in processed_nodes}

    assert (
        processed_texts == expected_texts
    ), "Nodes with repeated text above the threshold were not correctly removed."


### RemoveNodesBelowNTokens tests ###


def test_RemoveNodesBelowNTokens():
    sample_bbox = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}

    nodes = [
        create_text_node(text="Hello, world!" * 100, **sample_bbox),
        create_text_node(text="Hi!", **sample_bbox),  # This is a stub.
        create_text_node(text="Unique text" * 100, **sample_bbox),
        create_text_node(text="Bye", **sample_bbox),  # This is also a stub.
        create_text_node(text="Another unique text" * 100, **sample_bbox),
    ]

    processor = RemoveNodesBelowNTokens(min_tokens=50)
    processed_nodes = processor.process(nodes)

    expected_texts = {
        "Hello, world!" * 100,
        "Unique text" * 100,
        "Another unique text" * 100,
    }
    processed_texts = {node.text for node in processed_nodes}

    assert (
        processed_texts == expected_texts
    ), "Stubs were not correctly removed based on text length."


### CombineNodesSpatially tests ###


def test_combine_nodes_spatially_both_small():
    nodes = [
        create_text_node("Node 1", 0, 0, 100, 100),
        create_text_node("Node 2", 80, 80, 180, 180),
        create_text_node("Non-overlapping", 2000, 2000, 3000, 3000),
    ]

    processor = CombineNodesSpatially(
        x_error_margin=30, y_error_margin=30, criteria="both_small"
    )
    processed_nodes = processor.process(nodes)

    # Expect "Node 1" and "Node 2" to be combined due to overlap and both being small.
    # "Non-overlapping" remains as is because it does not overlap with the others.
    expected_texts = {
        "Node 1Node 2",
        "Non-overlapping",
    }
    processed_texts = {
        "".join(element.text for element in node.elements) for node in processed_nodes
    }

    assert len(processed_nodes) == 2, "Expected two nodes after combination."
    assert (
        processed_texts == expected_texts
    ), "Nodes were not combined correctly based on spatial and size criteria."


def test_combine_nodes_one_stub_one_small():
    nodes = [
        create_text_node("stub", 0, 0, 50, 50),  # Stub due to small area/content
        create_text_node("small " * 10, 40, 40, 140, 140),  # Small but not a stub
        create_text_node("non-overlapping " * 100, 300, 300, 400, 400),  # Separate node
    ]

    processor = CombineNodesSpatially(
        x_error_margin=30, y_error_margin=30, criteria="either_stub"
    )
    processed_nodes = processor.process(nodes)

    # Expect "Stub Node" and "Small Node" to be combined due to overlapping and at least one being a stub.
    # "Non-overlapping" remains separate.
    assert (
        len(processed_nodes) == 2
    ), "Expected 2 nodes after processing: one combined and one separate."

    # Assert that texts from the first two nodes are present in one of the processed nodes.
    # This checks for presence, not order or exact formatting.
    texts_from_combined_nodes = ["stub", "small " * 10]
    combined_text = " ".join(node.text for node in processed_nodes)

    for expected_text in texts_from_combined_nodes:
        assert (
            expected_text in combined_text
        ), f"Expected text from node not found in combined output: {expected_text}"


### CombineBullets tests ###


@pytest.fixture
def node_without_bullets():
    text_element = TextElement(
        text="Some text without bullet points.",
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
    return Node(elements=[text_element])


@pytest.fixture
def node_starts_with_bullet():
    text_element = TextElement(
        text=f"- Bullet point at the start.{consts.ELEMENT_DELIMETER}\n other text that follows.",
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
    return Node(elements=[text_element])


@pytest.fixture
def node_ends_with_bullet():
    text_element = TextElement(
        text=f"Text ending with a bullet point:\n{consts.ELEMENT_DELIMETER}- Bullet",
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
    return Node(elements=[text_element])


@pytest.fixture
def node_with_multiple_bullets():
    text_element = TextElement(
        text=f"- Bullet one{consts.ELEMENT_DELIMETER}- Bullet two{consts.ELEMENT_DELIMETER}- Bullet three",
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
    return Node(elements=[text_element])


def test_combine_bullets_single_node(node_without_bullets):
    processor = CombineBullets()
    result = processor.process([node_without_bullets])
    assert len(result) == 1, "Single node without bullets should not be modified."


def test_combine_bullets_sequential_bullets(
    node_starts_with_bullet, node_ends_with_bullet
):
    processor = CombineBullets()
    result = processor.process([node_ends_with_bullet, node_starts_with_bullet])
    assert len(result) == 1, "Nodes with sequential bullets should be combined."


def test_combine_bullets_no_combination(node_without_bullets, node_starts_with_bullet):
    processor = CombineBullets()
    result = processor.process([node_without_bullets, node_starts_with_bullet])
    assert len(result) == 2, "Nodes without sequential bullets should not be combined."


def test_combine_bullets_multiple_nodes(
    node_starts_with_bullet, node_with_multiple_bullets, node_ends_with_bullet
):
    processor = CombineBullets()
    result = processor.process(
        [node_starts_with_bullet, node_with_multiple_bullets, node_ends_with_bullet]
    )
    assert (
        len(result) == 3
    ), "All nodes with bullets should remain separate as they don't form a continuous list."


def test_combine_multiple_sequential_bullets(
    node_starts_with_bullet, node_ends_with_bullet
):
    processor = CombineBullets()
    # Simulating a scenario where multiple nodes with start/end bullets should be combined into one
    nodes = [
        node_ends_with_bullet,
        node_starts_with_bullet,
        node_ends_with_bullet,
        node_starts_with_bullet,
    ]
    result = processor.process(nodes)
    assert (
        len(result) == 2
    ), "Nodes with sequential start/end bullets should be combined accordingly."


### AppendShortNodesToPrevNodeWithHeading tests ###


@pytest.fixture
def heading_node():
    lines = LineElement(
        bbox=(0, 0, 100, 20),
        spans=[
            TextSpan(text="**Heading Example**", is_bold=True, size=18, is_italic=False)
        ],  # Assuming TextSpan is defined and relevant.
        style=None,
    )
    text_element = TextElement(
        text="**Heading Example**",
        lines=(lines,),
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
    return Node(elements=(text_element,), variant="text")


@pytest.fixture
def short_text_node():
    line_element = LineElement(
        bbox=(0, 30, 100, 50),
        spans=[
            TextSpan(
                text="This is some short text that follows a heading.",
                is_bold=False,
                is_italic=False,
                size=12,
            )
        ],
        style=None,
    )
    text_element = TextElement(
        text="This is some short text that follows a heading.",
        lines=(line_element,),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=679.1,
            x1=594.0,
            y1=731.3,
        ),
    )
    return Node(elements=(text_element,), variant="text")


@pytest.fixture
def longer_text_node():
    line_element = LineElement(
        bbox=(0, 60, 200, 100),
        spans=[
            TextSpan(
                text="This is a longer piece of text intended as content under a heading.",
                is_bold=False,
                is_italic=False,
                size=12,
            )
        ],
        style=None,
    )
    text_element = TextElement(
        text="This is a longer piece of text intended as content under a heading.",
        lines=(line_element,),
        bbox=Bbox(
            page=0,
            page_height=792.0,
            page_width=612.0,
            x0=18.22,
            y0=699.1,
            x1=594.0,
            y1=751.3,
        ),
    )
    node = Node(elements=(text_element,), variant="text")
    assert text_element.is_heading == False
    assert node.is_heading == False
    return node


def test_combine_heading_with_next_text(
    heading_node, short_text_node, longer_text_node
):
    processor = CombineHeadingsWithClosestText()
    nodes = [heading_node, short_text_node, longer_text_node]
    processed_nodes = processor.process(nodes)

    # Expect the heading node to combine with the short text node, leaving the longer text node separate.
    assert (
        len(processed_nodes) == 2
    ), "Expected two nodes after combining a heading with its closest text."

    combined_text = "".join(element.text for element in processed_nodes[0].elements)
    assert (
        "**Heading Example**" in combined_text
    ), "Heading should be part of the first combined node."
    assert (
        "This is some short text" in combined_text
    ), "Short text should be combined with the heading."

    remaining_text = "".join(element.text for element in processed_nodes[1].elements)
    assert (
        "This is a longer piece of text" in remaining_text
    ), "Longer text should remain in a separate node."


def test_no_combine_when_no_heading(short_text_node, longer_text_node):
    processor = CombineHeadingsWithClosestText()
    nodes = [short_text_node, longer_text_node]
    processed_nodes = processor.process(nodes)

    # Without a preceding heading node, no combination should occur.
    assert (
        len(processed_nodes) == 2
    ), "Nodes should remain separate when no heading is present."
