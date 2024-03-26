from typing import List, TypedDict, Union
from dataclasses import dataclass

from src.schemas import Node
from src.postprocessing.steps import PostProcessingStep, default_pipeline


class ProcessingArgs(TypedDict, total=False):
    min_tokens: int
    max_tokens: int
    processing_pipeline: List[PostProcessingStep]


@dataclass
class ParsedProcessingArgs:
    min_tokens: float = 128
    max_tokens: float = 1024
    processing_pipeline: List[PostProcessingStep] = default_pipeline


def merge_with_defaults(user_args: Union[ProcessingArgs, None]) -> ParsedProcessingArgs:
    args = ParsedProcessingArgs()

    if user_args:
        for key, value in user_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

    return args


def run_pipeline(nodes: List[Node], args: Union[ProcessingArgs, None]) -> List[Node]:
    parsed_args = merge_with_defaults(args)
    for transform in parsed_args.processing_pipeline:
        nodes = transform.process(nodes)
    return nodes
