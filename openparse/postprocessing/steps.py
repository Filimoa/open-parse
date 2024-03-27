from typing import Optional, List, Sequence, Literal, TypedDict, Union
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

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
