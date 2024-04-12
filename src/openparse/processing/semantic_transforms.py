import json

from typing import List, Literal, Dict, Union
from urllib.parse import urlparse
from http.client import HTTPConnection, HTTPSConnection

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
        except ImportError:
            raise ImportError(
                "You need to install the openai package to use this feature."
            )
        return OpenAI(api_key=self.api_key)


class OllamaEmbeddings:
    """
    Use local models via ollama for calculating embeddings. Uses the REST API
    https://github.com/ollama/ollama/blob/main/docs/api.md.

    * nomic-embed-text
    * mxbai-embed-large
    """

    def __init__(
        self,
        url: str = "http://localhost:11434/",
        model: str = "mxbai-embed-large",
        batch_size: int = 256,
    ):
        """
        Used to generate embeddings for Nodes.
        """
        self.url = url
        self.model = model
        self.batch_size = batch_size

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts. Support for batches coming
        soon, cf. https://ollama.com/blog/embedding-models

        Args:
            texts (list[str]): The list of texts to embed.
            batch_size (int): The number of texts to process in each batch.

        Returns:
            List[List[float]]: A list of embeddings.
        """
        conn = self._create_conn()
        res = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            for text in batch_texts:
                params = json.dumps({"model": self.model, "prompt": text})
                headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
                conn.request("POST", "/api/embeddings", params, headers)
                response = conn.getresponse()
                if response.status != 200:
                    raise RuntimeError(
                        "embeddings request failed: {} {}".format(
                            response.status, response.reason
                        )
                    )
                doc = json.loads(response.read())
                res.extend(doc["embedding"])
        conn.close()
        return res

    def _create_conn(self):
        parsed = urlparse(self.url)
        if parsed.scheme == "https":
            return HTTPSConnection(parsed.hostname, parsed.port)
        else:
            return HTTPConnection(parsed.hostname, parsed.port)


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
