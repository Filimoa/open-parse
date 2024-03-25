from typing import Optional, List, Sequence, Literal
from collections import defaultdict

from src.schemas import Node, TextElement, LineElement, Bbox


def remove_full_page_elements(
    nodes: List[Node],
    max_area_pct: float,
) -> List[TextElement]:
    raise NotImplementedError("Sub-elements are not yet supported.")
    res = []
    for e in elements:
        document_area = e.bbox.page_width * e.bbox.page_height

        if e.area / document_area < max_area_pct:
            res.append(e)
        elif not e.is_stub:
            res.append(e)

    return res


def remove_metadata_elements(
    nodes: list[Node],
    min_y0_pct: float = 0.12,
    max_y0_pct: float = 0.88,
) -> list[Node]:
    """
    Looking to remove page numbers, headers, footers, etc.
    """
    res = []
    for n in nodes:
        first_bbox = n.elements[0].bbox
        if n.num_pages > 1:
            res.append(n)
        elif (
            first_bbox.y0 > first_bbox.page_height * min_y0_pct
            and first_bbox.y1 < first_bbox.page_height * max_y0_pct
        ):
            res.append(n)
        elif not n.is_stub:
            res.append(n)
    return res


def remove_repeated_elements(nodes: list[Node], threshold: int = 2) -> list[Node]:
    text_counts: dict[str, int] = defaultdict(int)
    for node in nodes:
        if node.text is not None:
            text_counts[node.text] += 1

    repeated_texts = set(
        text for text, count in text_counts.items() if count > threshold
    )

    res = []
    for node in nodes:
        if node.text is None or node.text not in repeated_texts:
            res.append(node)

    return res


def remove_stubs(elements: list[Node]) -> list[Node]:
    return [e for e in elements if not e.is_stub]


def _combine_nodes_spatially(
    nodes: List[Node],
    x_error_margin: float = 0,
    y_error_margin: float = 0,
    criteria: Literal["both_small", "either_stub"] = "both_small",
) -> List[Node]:
    # Initially, no nodes are combined.
    combined_nodes = []

    while nodes:
        current_node = nodes.pop(0)
        combined = False

        for i, target_node in enumerate(combined_nodes):
            if criteria == "both_small":
                criteria_bool = current_node.is_small and target_node.is_small
            elif criteria == "either_stub":
                criteria_bool = current_node.is_stub or target_node.is_stub

            if (
                current_node.overlaps(
                    target_node,
                    x_error_margin=x_error_margin,
                    y_error_margin=y_error_margin,
                )
                and criteria_bool
            ):
                new_elements = target_node.elements + current_node.elements
                combined_nodes[i] = Node(elements=new_elements)
                combined = True
                break

        if not combined:
            combined_nodes.append(current_node)

    return combined_nodes


###############
# OLD STUFF


# def _split_large_elements(elements: list[TextElement]) -> list[TextElement]:
#     PIXELS_IN_TAB = 2
#     STUB_LENGTH_PCT = 0.3
#     res = []
#     for element in elements:
#         if not element.is_large:
#             res.append(element)
#             continue

#         median_x0 = statistics.median([l.x0 for l in element.lines])
#         median_len = statistics.median([len(l.text) for l in element.lines])
#         split_points = []

#         for line in element.lines:
#             if line.x0 > median_x0 + PIXELS_IN_TAB:
#                 split_points.append(line.y0)
#             elif len(line.text) < median_len * STUB_LENGTH_PCT:
#                 split_points.append(line.y1)

#         if len(split_points) == 0:
#             res.append(element)
#             continue

#         res.extend(element.split(split_points))
#     return res


def _combine_text_and_table_elements(
    text_elements: list[TextElement], table_elements: list[TableElement]
) -> Sequence[TextElement | TableElement]:
    """Combines text and table elements into a single list. Remove any text elements that are found on a page with a table (for now)"""
    filtered_text_elems: list[TextElement] = []
    pages_with_tables = {table.page for table in table_elements}
    for element in text_elements:
        if element.page not in pages_with_tables:
            filtered_text_elems.append(element)
    return filtered_text_elems + table_elements  # type: ignore


def _transform_elements(
    pdf: fitz.Document, summarize_tables: bool = False
) -> Sequence[TextElement | TableElement]:
    document_area = pdf[0].rect.width * pdf[0].rect.height

    res = _parse_elements(pdf)
    res = remove_full_page_elements(res, document_area)

    # intial merging
    res = _combine_elements_spatially(res, x_error_margin=4, y_error_margin=4)
    res = _combine_elements_spatially(res)
    res = _combine_bullets(res)

    # dealing with stubs
    res = _remove_metadata_elements(res, pdf[0].rect.height)
    res = _combine_elements_spatially(
        res, x_error_margin=4, y_error_margin=12, critera="either_stub"
    )
    res = _combine_elements_spatially(res, critera="either_stub")
    res = remove_stubs(res)

    # misc cleanup
    res = _split_large_elements(res)
    res = remove_repeated_elements(res)
    res = _standardize_positions(res)

    # parse tables
    res_with_tables: Sequence[TextElement | TableElement] = res
    parsed_tables = []
    if pdf.page_count > MAX_PAGES_FOR_TABLES and summarize_tables:
        logger.info(f"Skipping table parsing due to page count {pdf.page_count}")
    else:
        parsed_tables = asyncio.run(
            tables.parse_tables(pdf, generate_summary=summarize_tables)
        )

    res_with_tables = _combine_text_and_table_elements(res, parsed_tables)

    res_with_tables = _standardize_positions(res_with_tables)

    return res_with_tables
