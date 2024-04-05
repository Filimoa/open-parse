from typing import List, Optional

from openparse.processing.steps import default_pipeline, ProcessingStep
from openparse.schemas import Node


def run_pipeline(
    nodes: List[Node],
    pipeline: List[ProcessingStep],
    verbose: bool = False,
) -> List[Node]:
    for transform_func in pipeline:
        if verbose:
            print("Processing with", transform_func.__class__.__name__)
        nodes = transform_func.process(nodes)
        nodes = sorted(nodes, key=lambda x: x.aggregate_position, reverse=True)
    return nodes
