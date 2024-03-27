from abc import ABC, abstractmethod
from typing import List

from openparse.schemas import Node


class PostProcessingStep(ABC):
    @abstractmethod
    def process(self, nodes: List[Node]) -> List[Node]:
        """
        Process a list of Nodes and return a modified list of Nodes.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class CombineNodesSplitAcrossPages(PostProcessingStep):
    def __init__(self):
        pass

    def process(self, nodes: List[Node]) -> List[Node]:
        raise NotImplementedError("CombineNodesSplitAcrossPages not implemented")


default_pipeline = [
    CombineNodesSplitAcrossPages(),
    # CombineBullets(),
    # CombineHeadingsWithClosestText(),
]
