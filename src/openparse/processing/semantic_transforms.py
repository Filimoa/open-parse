from abc import ABC, abstractmethod
from typing import List, Literal, Union

import numpy as np

from openparse.schemas import Node

from .basic_transforms import ProcessingStep

EmbeddingModel = Literal[
    "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"
]


def cosine_similarity(
    a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space that measures the cosine of the angle between them.

    Parameters:
    a (Union[np.ndarray, List[float]]): The first vector.
    b (Union[np.ndarray, List[float]]): The second vector.

    Returns:
    float: The cosine similarity between vector `a` and vector `b`. The value ranges from -1 meaning exactly opposite, to 1 meaning exactly the same, with 0 usually indicating orthogonality (independence), and in-between values indicating intermediate similarity or dissimilarity.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class BaseEmbeddings(ABC):
    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(
        self,
        model: EmbeddingModel,
        api_key: str,
        batch_size: int = 256,
    ):
        """
        Used to generate embeddings for Nodes.

        Args:
            api_key (str): Your OpenAI API key.
            model (str): The embedding model to use.
            batch_size (int): The number of texts to process in each api call.
        """
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.client = self._create_client()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        res = []
        non_empty_texts = [text for text in texts if text]

        embedding_size = 1
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch_texts = non_empty_texts[i : i + self.batch_size]
            api_resp = self.client.embeddings.create(
                input=batch_texts, model=self.model
            )
            batch_res = [val.embedding for val in api_resp.data]
            res.extend(batch_res)
            embedding_size = len(batch_res[0])

        # Map results back to original indices, adding zero embeddings for empty texts
        final_res = [
            [0.0] * embedding_size if not text else res.pop(0) for text in texts
        ]

        return final_res

    def _create_client(self):
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "You need to install the openai package to use this feature."
            ) from err
        return OpenAI(api_key=self.api_key)

class AzureOpenAIEmbeddings(BaseEmbeddings):
    def __init__(
        self,
        api_key: str,
        api_endpoint: str,
        deployment: str,
        api_version: str = "2024-02-15-preview",
        batch_size: int = 256,
    ):
        """
        Used to generate embeddings for Nodes.

        Args:
            model (str): The embedding model to use.
            api_key (str): Your Azure OpenAI API key.
            api_endpoint (str): The Azure endpoint to use.
            api_version (str): The version of the API to use.
            deployment (str): The deployment to use.
            batch_size (int): The number of texts to process in each api call.
        """
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.api_version = api_version
        self.deployment = deployment
        self.batch_size = batch_size
        self.client = self._create_client()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts in batches.

        Args:
            texts (list[str]): The list of texts to embed.
            batch_size (int): The number of texts to process in each batch.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        res = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            api_resp = self.client.embeddings.create(
                input=batch_texts, model=self.deployment
            )
            batch_res = [val.embedding for val in api_resp.data]
            res.extend(batch_res)

        return res

    def _create_client(self):
        try:
            from openai import AzureOpenAI
        except ImportError as err:
            raise ImportError(
                "You need to install the openai package to use this feature."
            ) from err
        return AzureOpenAI(api_key=self.api_key, azure_endpoint=self.api_endpoint, azure_deployment=self.deployment, api_version=self.api_version)
    
class CombineNodesSemantically(ProcessingStep):
    """
    Combines nodes that are semantically related.
    """

    def __init__(
        self,
        embedding_client: BaseEmbeddings,
        min_similarity: float,
        max_tokens: int,
    ):
        self.embedding_client = embedding_client
        self.min_similarity = min_similarity
        self.max_tokens = max_tokens

    def process(self, nodes: List[Node]) -> List[Node]:
        modified = True
        while modified:
            modified = False
            nodes = sorted(nodes)

            embeddings = self.embedding_client.embed_many([node.text for node in nodes])
            i = 0

            while i < len(nodes) - 1:
                current_embedding = embeddings[i]
                next_embedding = embeddings[i + 1]
                similarity = cosine_similarity(current_embedding, next_embedding)
                is_within_token_limit = (
                    nodes[i].tokens + nodes[i + 1].tokens <= self.max_tokens
                )

                if similarity >= self.min_similarity and is_within_token_limit:
                    nodes[i] = nodes[i] + nodes[i + 1]
                    del nodes[i + 1]
                    del embeddings[i + 1]

                    modified = True
                    continue
                i += 1

        return nodes

    def _get_node_similarities(self, nodes: List[Node]) -> List[float]:
        """
        Get the similarity of each node with the node that precedes it
        """
        embeddings = self.embedding_client.embed_many([node.text for node in nodes])

        similarities = []
        for i in range(1, len(embeddings)):
            similarities.append(cosine_similarity(embeddings[i - 1], embeddings[i]))

        return [0] + similarities
