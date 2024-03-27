from typing import List, TypedDict, Union

from src.schemas import Node
from src.processing.steps import ProcessingStep, default_pipeline


def run_pipeline(nodes: List[Node], args: Union[dict, None] = None) -> List[Node]:
    for transform in default_pipeline:
        nodes = transform.process(nodes)
    return nodes
