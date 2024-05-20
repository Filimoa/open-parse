"""
This is meant to provide a simple wrapper around llama_index's embeddings classes.
"""

from typing import Dict, Type

from llama_index.core.embeddings import BaseEmbedding


class ImportErrorProxy:
    """
    Used to raise an ImportError when an attribute or method is accessed on a class that failed to import.
    """

    def __init__(self, class_name, install_command):
        self.class_name = class_name
        self.install_command = install_command
        self.error_message = (
            f"Missing optional dependency for '{class_name}'. "
            f"Please install it by running: '{install_command}'."
        )

    def __getattr__(self, name):
        raise ImportError(
            f"{self.error_message} The attribute '{name}' cannot be used."
        )

    def __call__(self, *args, **kwargs):
        raise ImportError(self.error_message)


try:
    from llama_index.embeddings.openai import (
        OpenAIEmbedding,
    )

except ImportError:
    OpenAIEmbedding = ImportErrorProxy(
        "OpenAIEmbedding",
        "pip install openparse[embeddings-openai]",
    )

try:
    from llama_index.embeddings.azure_openai import (
        AzureOpenAIEmbedding,
    )

except ImportError:
    AzureOpenAIEmbedding = ImportErrorProxy(
        "AzureOpenAIEmbedding",
        "pip install openparse[embeddings-azure-openai]",
    )

try:
    from llama_index.embeddings.huggingface import (
        HuggingFaceInferenceAPIEmbedding,
    )

except ImportError:
    HuggingFaceInferenceAPIEmbedding = ImportErrorProxy(
        "HuggingFaceInferenceAPIEmbedding",
        "pip install openparse[embeddings-huggingface]",
    )


try:
    from llama_index.embeddings.huggingface_optimum import (
        OptimumEmbedding,
    )

except ImportError:
    OptimumEmbedding = ImportErrorProxy(
        "OptimumEmbedding",
        "pip install openparse[embeddings-huggingface-optimum]",
    )

try:
    from llama_index.embeddings.cohere import CohereEmbedding

except ImportError:
    CohereEmbedding = ImportErrorProxy(
        "CohereEmbedding",
        "pip install openparse[embeddings-cohere]",
    )


try:
    from llama_index.embeddings.text_embeddings_inference import (
        TextEmbeddingsInference,
    )

except ImportError:
    TextEmbeddingsInference = ImportErrorProxy(
        "TextEmbeddingsInference",
        "pip install openparse[embeddings-text-embeddings-inference]",
    )
