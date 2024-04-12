from typing import List, Optional
from abc import ABC

from openparse.schemas import Node
from openparse import consts
from openparse.processing.basic_transforms import (
    ProcessingStep,
    RemoveTextInsideTables,
    RemoveFullPageStubs,
    CombineNodesSpatially,
    CombineHeadingsWithClosestText,
    CombineBullets,
    RemoveMetadataElements,
    RemoveRepeatedElements,
    RemoveNodesBelowNTokens,
)
from openparse.processing.semantic_transforms import (
    CombineNodesSemantically,
    OpenAIEmbeddings,
    OllamaEmbeddings,
    EmbeddingModel,
)


class IngestionPipeline(ABC):
    """
    A pipeline for ingesting and processing Nodes.

    Attributes:
        transformations (List[ProcessingStep]): A list of transforms to process the extracted elements.
        verbose (Optional[bool]): Whether to print out processing steps.
    """

    transformations: List[ProcessingStep]
    verbose: Optional[bool] = False

    def run(self, nodes: List[Node]) -> List[Node]:
        nodes = sorted(nodes)
        for transform_func in self.transformations:
            if self.verbose:
                print("Processing with", transform_func.__class__.__name__)
            nodes = transform_func.process(sorted(nodes))

        return nodes

    def append_transform(self, transform: ProcessingStep) -> None:
        """
        Add a transform to the pipeline.

        Args:
            transform (ProcessingStep): The transform to add.
        """
        self.transformations.append(transform)


class NoOpIngestionPipeline(IngestionPipeline):
    """
    A no-operation (no-op) pipeline for cases where no processing should be performed.
    """

    def __init__(self):
        self.transformations = []


class BasicIngestionPipeline(IngestionPipeline):
    """
    A basic pipeline for ingesting and processing Nodes.
    """

    def __init__(self):
        self.transformations = [
            RemoveTextInsideTables(),
            RemoveFullPageStubs(max_area_pct=0.35),
            # mostly aimed at combining bullets and weird formatting
            CombineNodesSpatially(
                x_error_margin=10, y_error_margin=4, criteria="both_small"
            ),
            CombineHeadingsWithClosestText(),
            CombineBullets(),
            CombineNodesSpatially(
                x_error_margin=0, y_error_margin=10, criteria="both_small"
            ),
            RemoveMetadataElements(),
            CombineNodesSpatially(criteria="either_stub"),
            RemoveRepeatedElements(threshold=2),
            # # tried everything to combine, remove stubs that are still left
            RemoveNodesBelowNTokens(min_tokens=50),
            # # combines bullets split across pages
            # # (previously page metdata would have prevented this)
            CombineBullets(),
        ]


class SemanticIngestionPipeline(IngestionPipeline):
    """
    A semantic pipeline for ingesting and processing Nodes.
    """

    def __init__(
        self,
        openai_api_key: str,
        model: EmbeddingModel = "text-embedding-3-large",
        min_tokens: int = consts.TOKENIZATION_LOWER_LIMIT,
        max_tokens: int = consts.TOKENIZATION_UPPER_LIMIT,
    ) -> None:
        embedding_client = OpenAIEmbeddings(api_key=openai_api_key, model=model)

        self.transformations = [
            RemoveTextInsideTables(),
            RemoveFullPageStubs(max_area_pct=0.35),
            # mostly aimed at combining bullets and weird formatting
            CombineNodesSpatially(
                x_error_margin=10,
                y_error_margin=2,
                criteria="both_small",
            ),
            CombineHeadingsWithClosestText(),
            CombineBullets(),
            RemoveMetadataElements(),
            RemoveRepeatedElements(threshold=2),
            RemoveNodesBelowNTokens(min_tokens=10),
            CombineBullets(),
            CombineNodesSemantically(
                embedding_client=embedding_client,
                min_similarity=0.6,
                max_tokens=max_tokens // 2,
            ),
            CombineNodesSemantically(
                embedding_client=embedding_client,
                min_similarity=0.55,
                max_tokens=max_tokens,
            ),
            RemoveNodesBelowNTokens(min_tokens=min_tokens),
        ]

class LocalSemanticIngestionPipeline(IngestionPipeline):
    """
    A semantic pipeline for ingesting and processing Nodes using ollama for embeddings.
    """

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "mxbai-embed-large",
        min_tokens: int = consts.TOKENIZATION_LOWER_LIMIT,
        max_tokens: int = consts.TOKENIZATION_UPPER_LIMIT,
    ) -> None:
        embedding_client = OllamaEmbeddings(url=url, model=model)

        self.transformations = [
            RemoveTextInsideTables(),
            RemoveFullPageStubs(max_area_pct=0.35),
            # mostly aimed at combining bullets and weird formatting
            CombineNodesSpatially(
                x_error_margin=10,
                y_error_margin=2,
                criteria="both_small",
            ),
            CombineHeadingsWithClosestText(),
            CombineBullets(),
            RemoveMetadataElements(),
            RemoveRepeatedElements(threshold=2),
            RemoveNodesBelowNTokens(min_tokens=10),
            CombineBullets(),
            CombineNodesSemantically(
                embedding_client=embedding_client,
                min_similarity=0.6,
                max_tokens=max_tokens // 2,
            ),
            CombineNodesSemantically(
                embedding_client=embedding_client,
                min_similarity=0.55,
                max_tokens=max_tokens,
            ),
            RemoveNodesBelowNTokens(min_tokens=min_tokens),
        ]
