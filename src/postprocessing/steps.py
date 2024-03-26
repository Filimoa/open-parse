from typing import Optional, List, Sequence, Literal, TypedDict, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

from src.schemas import Node


class ProcessingStep(ABC):
    @abstractmethod
    def process(self, nodes: List[Node]) -> List[Node]:
        """
        Process a list of Nodes and return a modified list of Nodes.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CombineNodesSplitAcrossPages(ProcessingStep):
    def __init__(self, openai_client: Optional[] = None):
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


default_pipeline = [
    CombineNodesSplitAcrossPages(),
    # CombineBullets(),
    # CombineHeadingsWithClosestText(),
]
