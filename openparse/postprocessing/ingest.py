from typing import List, TypedDict, Union
from dataclasses import dataclass

from openparse.schemas import Node
from openparse.postprocessing.steps import PostProcessingStep, default_pipeline


class ProcessingArgs(TypedDict, total=False):
    min_tokens: int
    max_tokens: int
    processing_pipeline: List[PostProcessingStep]


def run_pipeline(nodes: List[Node], args: Union[dict, None] = None) -> List[Node]:
    for transform in default_pipeline:
        nodes = transform.process(nodes)
    return nodes
