from openparse.processing import (
    ProcessingStep,
    RemoveTextInsideTables,
    RemoveRepeatedElements,
    RemoveFullPageStubs,
    RemoveStubs,
    CombineNodesSpatially,
)
from openparse.schemas import Node, TextElement, TableElement, Bbox


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


### RemoveStubs tests ###


def test_remove_stubs():
    sample_bbox = {"x0": 0, "y0": 0, "x1": 10, "y1": 10}

    nodes = [
        create_text_node(text="Hello, world!" * 100, **sample_bbox),
        create_text_node(text="Hi!", **sample_bbox),  # This is a stub.
        create_text_node(text="Unique text" * 100, **sample_bbox),
        create_text_node(text="Bye", **sample_bbox),  # This is also a stub.
        create_text_node(text="Another unique text" * 100, **sample_bbox),
    ]

    processor = RemoveStubs()
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

### RemoveStubs tests ###


# def create_combined_text_node(texts, x0, y0, x1, y1, page=1):
#     """Helper function to create a Node with combined text elements."""
#     elements = tuple(
#         TextElement(
#             text=text,
#             lines=(),
#             bbox=Bbox(
#                 x0=x0, y0=y0, x1=x1, y1=y1, page=page, page_height=5000, page_width=5000
#             ),
#         )
#         for text in texts
#     )
#     return Node(elements=elements)


# def test_combine_nodes_spatially_both_small():
#     nodes = [
#         TestNode(
#             create_text_node("Node 1", 0, 0, 100, 100), is_stub=False, is_small=True
#         ),
#         TestNode(
#             create_text_node("Node 2", 80, 80, 180, 180), is_stub=False, is_small=True
#         ),
#         TestNode(
#             create_text_node("Non-overlapping", 200, 200, 300, 300),
#             is_stub=False,
#             is_small=True,
#         ),
#     ]

#     processor = CombineNodesSpatially(
#         x_error_margin=50, y_error_margin=50, criteria="both_small"
#     )
#     processed_nodes = processor.process(nodes)

#     # Expect "Node 1" and "Node 2" to be combined due to overlap and both being small.
#     # "Non-overlapping" remains as is because it does not overlap with the others.
#     expected_texts = {
#         "Node 1Node 2",
#         "Non-overlapping",
#     }
#     processed_texts = {
#         "".join(element.text for element in node.elements) for node in processed_nodes
#     }

#     assert len(processed_nodes) == 2, "Expected two nodes after combination."
#     assert (
#         processed_texts == expected_texts
#     ), "Nodes were not combined correctly based on spatial and size criteria."
