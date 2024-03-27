from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Literal, Dict

from openparse.schemas import Bbox, Node, TextElement


class ProcessingStep(ABC):
    @abstractmethod
    def process(self, nodes: List[Node]) -> List[Node]:
        """
        Process a list of Nodes and return a modified list of Nodes.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class RemoveTextInsideTables(ProcessingStep):
    def process(self, nodes: List[Node]) -> List[Node]:
        # Group all table bounding boxes by page
        tables_by_page = defaultdict(list)
        for node in nodes:
            if node.variant == "table":
                for table_element in node.elements:
                    tables_by_page[table_element.page].append(table_element.bbox)

        updated_nodes = []
        for node in nodes:
            if node.variant == "table":
                updated_nodes.append(node)
                continue

            new_elements = [
                element
                for element in node.elements
                if not (
                    isinstance(element, TextElement)
                    and self.is_inside_any_table(
                        element.bbox, tables_by_page[element.page]
                    )
                )
            ]

            if new_elements:
                updated_nodes.append(Node(elements=tuple(new_elements)))

        return updated_nodes

    def is_inside_any_table(self, text_bbox: Bbox, table_bboxes: List[Bbox]) -> bool:
        return any(
            self.is_contained(text_bbox, table_bbox) for table_bbox in table_bboxes
        )

    @staticmethod
    def is_contained(text_bbox: Bbox, table_bbox: Bbox) -> bool:
        return (
            text_bbox.x0 >= table_bbox.x0
            and text_bbox.x1 <= table_bbox.x1
            and text_bbox.y0 >= table_bbox.y0
            and text_bbox.y1 <= table_bbox.y1
        )


class RemoveFullPageStubs(ProcessingStep):
    def __init__(self, max_area_pct: float):
        self.max_area_pct = max_area_pct

    def process(self, nodes: List[Node]) -> List[Node]:
        res = []
        for node in nodes:
            node_bbox = node.bbox[0]
            page_area = node_bbox.page_width * node_bbox.page_height

            if node.num_pages > 1:
                res.append(node)
                continue
            elif node_bbox.area / page_area < self.max_area_pct:
                res.append(node)
                continue
            elif not node.is_stub:
                res.append(node)
                continue
        return res


class RemoveMetadataElements(ProcessingStep):
    def __init__(self, min_y0_pct: float = 0.1, max_y0_pct: float = 0.90):
        self.min_y0_pct = min_y0_pct
        self.max_y0_pct = max_y0_pct

    def process(self, nodes: List[Node]) -> List[Node]:
        res = []
        for node in nodes:
            if not node.elements:
                continue
            first_bbox = node.elements[0].bbox
            last_bbox = node.elements[-1].bbox

            # ignoring multi-page elements
            if first_bbox.page != last_bbox.page:
                continue

            is_within_allowed_range = (
                first_bbox.y0 >= first_bbox.page_height * self.min_y0_pct
                and last_bbox.y1 <= first_bbox.page_height * self.max_y0_pct
            )

            if is_within_allowed_range or not node.is_stub:
                res.append(node)
        return res


class RemoveRepeatedElements(ProcessingStep):
    def __init__(self, threshold: int = 2):
        self.threshold = threshold

    def process(self, nodes: List[Node]) -> List[Node]:
        text_counts: Dict[str, int] = defaultdict(int)
        for node in nodes:
            if node.text:
                text_counts[node.text] += 1

        repeated_texts = {
            text for text, count in text_counts.items() if count > self.threshold
        }

        return [
            node for node in nodes if not node.text or node.text not in repeated_texts
        ]


class RemoveStubs(ProcessingStep):
    def process(self, nodes: List[Node]) -> List[Node]:
        return [node for node in nodes if not node.is_stub]


class CombineNodesSpatially(ProcessingStep):
    def __init__(
        self,
        x_error_margin: float = 0,
        y_error_margin: float = 0,
        criteria: Literal["both_small", "either_stub"] = "both_small",
    ):
        self.x_error_margin = x_error_margin
        self.y_error_margin = y_error_margin
        self.criteria = criteria

    def process(self, nodes: List[Node]) -> List[Node]:
        combined_nodes: List[Node] = []

        while nodes:
            current_node = nodes.pop(0)
            combined = False

            for i, target_node in enumerate(combined_nodes):
                criteria_bool = False
                if self.criteria == "both_small":
                    criteria_bool = current_node.is_small and target_node.is_small
                elif self.criteria == "either_stub":
                    criteria_bool = current_node.is_stub or target_node.is_stub

                if (
                    current_node.overlaps(
                        target_node, self.x_error_margin, self.y_error_margin
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


class CombineBullets(ProcessingStep):
    def process(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError()


class CombineHeadingsWithClosestText(ProcessingStep):
    def process(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError()


# optimzed for pdfminer
default_pipeline = [
    RemoveTextInsideTables(),
    RemoveFullPageStubs(max_area_pct=0.5),
    CombineNodesSpatially(x_error_margin=10, y_error_margin=4, criteria="both_small"),
    CombineNodesSpatially(x_error_margin=0, y_error_margin=10, criteria="both_small"),
    # CombineBullets(),
    RemoveMetadataElements(),
    CombineNodesSpatially(criteria="either_stub"),
    # SplitLargeElements(),  # Implement
    RemoveRepeatedElements(threshold=2),
    RemoveStubs(),
    # CombineHeadingsWithClosestText(),  # Implement
]
