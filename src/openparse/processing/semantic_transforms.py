from typing import List, Literal, Dict, Union

import numpy as np

from openparse.schemas import Node
from .basic_transforms import ProcessingStep

EmbeddingModel = Literal[
    "text-embedding-3-large", "text-embedding-3-small", "text-embedding-ada-002"
]


def cosine_similarity(
    a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class OpenAIEmbeddings:
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
                input=batch_texts, model=self.model
            )
            batch_res = [val.embedding for val in api_resp.data]
            res.extend(batch_res)

        return res

    def _create_client(self):
        try:
            from openai import OpenAI
        except ImportError as err:
            raise ImportError(
                "You need to install the openai package to use this feature."
            ) from err
        return OpenAI(api_key=self.api_key)


class CombineNodesSemantically(ProcessingStep):
    """
    Combines nodes that are semantically related.
    """

    def __init__(
        self,
        embedding_client: OpenAIEmbeddings,
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
