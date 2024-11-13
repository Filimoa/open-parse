import base64
import io
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Literal, Type, TypeVar

from PIL import Image

from openparse.schemas import Bbox, ImageElement, Node, TableElement, TextElement

E = TypeVar("E", TextElement, ImageElement, TableElement)


def get_elements_of_type(nodes: List[Node], element_type: Type[E]) -> List[E]:
    elements: List[E] = []
    for node in nodes:
        for element in node.elements:
            if isinstance(element, element_type):
                elements.append(element)
    return elements


class ProcessingStep(ABC):
    @abstractmethod
    def process(self, nodes: List[Node]) -> List[Node]:
        """
        Process a list of Nodes and return a modified list of Nodes.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CombineSlicedImages(ProcessingStep):
    """
    PDF will slice images into multiple pieces if they are too large. This combines them back together.
    """

    def _combine_images_in_group(
        self, image_elements: List[ImageElement]
    ) -> ImageElement:
        """Combine a list of ImageElements into a single ImageElement."""
        if not image_elements:
            raise ValueError("No images to combine.")

        images = []
        for node in image_elements:
            image_data = base64.b64decode(node.image)
            image = Image.open(io.BytesIO(image_data))
            # image = image.rotate(180)
            images.append(image)

        # Determine the width and total height of the final image
        width = max(img.width for img in images)
        total_height = sum(img.height for img in images)

        # Create a new blank image
        new_image = Image.new("RGB", (width, total_height))

        # Paste images one below the other
        y_offset = 0
        for img in images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Save or encode the final image
        buffered = io.BytesIO()
        new_image.save(buffered, format="PNG")
        final_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return ImageElement(
            bbox=image_elements[0].bbox,
            image=final_base64,
            image_mimetype="image/png",
            text="",
        )

    def _group_overlapping_images(
        self, image_elements: List[ImageElement], buffer: float = 1.0
    ) -> List[List[ImageElement]]:
        """Group images that overlap or are adjacent."""
        groups = []
        used = set()

        for i, elem1 in enumerate(image_elements):
            if i in used:
                continue
            group = [elem1]
            used.add(i)
            queue = [elem1]
            while queue:
                current = queue.pop()
                for j, elem2 in enumerate(image_elements):
                    if j in used:
                        continue
                    if current.overlaps(elem2, buffer=buffer):
                        group.append(elem2)
                        used.add(j)
                        queue.append(elem2)
            groups.append(group)
        return groups

    def process(self, nodes: List[Node]) -> List[Node]:
        nodes_by_page: Dict[int, List[Node]] = defaultdict(list)
        for node in nodes:
            pages = {element.bbox.page for element in node.elements}
            for page in pages:
                nodes_by_page[page].append(node)

        new_nodes = []
        for page, page_nodes in nodes_by_page.items():
            image_nodes = [e for e in page_nodes if e.variant == {"image"}]
            if image_nodes:
                image_elements = get_elements_of_type(image_nodes, ImageElement)
                text_elements = get_elements_of_type(page_nodes, TextElement)

                combined_image = self._combine_images_in_group(image_elements)
                new_nodes.append(Node(elements=(combined_image, *text_elements)))
            else:
                new_nodes.extend(page_nodes)
        return new_nodes


class RemoveTextInsideTables(ProcessingStep):
    """
    If we're using the table extraction pipeline, we need to remove text that is inside tables to avoid duplication.
    """

    def process(self, nodes: List[Node]) -> List[Node]:
        # Group all table bounding boxes by page
        tables_by_page = defaultdict(list)
        for node in nodes:
            if node.variant == {"table"}:
                for table_element in node.elements:
                    tables_by_page[table_element.page].append(table_element.bbox)

        updated_nodes = []
        for node in nodes:
            if node.variant == {"table"}:
                updated_nodes.append(node)
                continue

            new_elements = []
            for element in node.elements:
                should_include = not (
                    isinstance(element, TextElement)
                    and self.intersects_any_table(
                        element.bbox, tables_by_page[element.page]
                    )
                )
                if should_include:
                    new_elements.append(element)

            if new_elements and len(new_elements) != len(node.elements):
                updated_nodes.append(Node(elements=tuple(new_elements)))
            elif len(new_elements) == len(node.elements):
                updated_nodes.append(node)

        return updated_nodes

    def intersects_any_table(self, text_bbox: Bbox, table_bboxes: List[Bbox]) -> bool:
        return any(
            self.intersects(text_bbox, table_bbox) for table_bbox in table_bboxes
        )

    @staticmethod
    def intersects(text_bbox: Bbox, table_bbox: Bbox) -> bool:
        return (
            text_bbox.x1 > table_bbox.x0
            and text_bbox.x0 < table_bbox.x1
            and text_bbox.y1 > table_bbox.y0
            and text_bbox.y0 < table_bbox.y1
        )


class RemoveFullPageStubs(ProcessingStep):
    """
    Sometimes elements take up entire pages and are not useful for downstream processing.
    """

    def __init__(self, max_area_pct: float):
        assert 0 <= max_area_pct <= 1, "max_area_pct must be between 0 and 1."
        self.max_area_pct = max_area_pct

    def process(self, nodes: List[Node]) -> List[Node]:
        """
        Retains multi-page nodes, nodes that occupy less than max_area_pct of the page.
        """
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
    """
    Looking to remove `page`, `attachment` etc. from the extracted text.  Typically we find this data to be quite challenging to incorporate into the querying stage ("tell me what's on page 6") without adding a lot of complexity to your app.
    """

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
    """
    Designed to remove repeated elements, such as headers and footers.
    This should be one of the last steps in the pipeline since we want to do everything possible to try to combine this with something else. Only if we can't combine it with anything else should we remove it.

    Note duplicates get droppped entirely, not just one of them. This is because typically the data is just metadata and not useful.
    """

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


class RemoveNodesBelowNTokens(ProcessingStep):
    """
    This should be the last step in the pipeline. Stubs are typically small elements that are not useful for downstream processing.
    """

    def __init__(self, min_tokens: int):
        self.min_tokens = min_tokens

    def process(self, nodes: List[Node]) -> List[Node]:
        res = []
        for node in nodes:
            if node.tokens <= self.min_tokens and "image" not in node.variant:
                continue
            res.append(node)
        return res


class CombineNodesSpatially(ProcessingStep):
    """
    Combines nodes that are close to each other spatially. We assume that elements that are close together on the page are related to each other and therefore should be combined.

    This simple heuristic achieves results comparable to deep learning methods we've experimented with. It's also much faster and easier to understand.

    Criteria:
    - both_small: Both nodes must be small elements. This is useful for combining small text elements that are close together. Common example is a bulleted list.
    - either_stub: Either node can be a stub. This is useful for combining small text elements like a heading with a larger text element below it.
    """

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
    """
    Needs to follow CombineNodesSpatially in the pipeline. Bullets are often far enough from the text they belong to that they are in distinct nodes.
    """

    def process(self, nodes: List[Node]) -> List[Node]:
        combined_nodes = []
        i = 0
        while i < len(nodes):
            current_combination = nodes[i]
            while (
                i + 1 < len(nodes)
                and current_combination.ends_with_bullet
                and nodes[i + 1].starts_with_bullet
            ):
                current_combination += nodes[i + 1]
                i += 1
            combined_nodes.append(current_combination)
            i += 1
        return combined_nodes


class CombineHeadingsWithClosestText(ProcessingStep):
    def process(self, nodes: List[Node]) -> List[Node]:
        res = []
        i = 0

        while i < len(nodes) - 1:
            current_node = nodes[i]

            if current_node.is_heading:
                next_node = nodes[i + 1]

                if not next_node.is_heading:
                    combined_node = current_node + next_node
                    res.append(combined_node)

                    # Skip the next node since it's been combined
                    i += 2
                    continue

            res.append(current_node)
            i += 1

        if i == len(nodes) - 1:
            res.append(nodes[i])

        return res
