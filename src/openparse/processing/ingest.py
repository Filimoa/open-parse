from typing import List, Union

from openparse.processing.steps import default_pipeline
from openparse.schemas import Node


def run_pipeline(nodes: List[Node], args: Union[dict, None] = None) -> List[Node]:
    for transform in default_pipeline:
        nodes = transform.process(nodes)
    return nodes
