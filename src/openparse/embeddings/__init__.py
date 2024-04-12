"""
This is meant to provide a simple wrapper around llama_index's embeddings classes.
"""

from typing import Dict, Type


class ImportErrorProxy:
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


RECOGNIZED_EMBEDDINGS: Dict[str, Type[int]] = {}

try:
    from llama_index.embeddings.openai import (
        OpenAIEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[OpenAIEmbedding.class_name()] = OpenAIEmbedding
except ImportError:
    OpenAIEmbedding = ImportErrorProxy(
        "OpenAIEmbedding",
        "pip install openparse[embeddings-openai]",
    )

try:
    from llama_index.embeddings.azure_openai import (
        AzureOpenAIEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[AzureOpenAIEmbedding.class_name()] = AzureOpenAIEmbedding
except ImportError:
    AzureOpenAIEmbedding = ImportErrorProxy(
        "AzureOpenAIEmbedding",
        "pip install openparse[embeddings-azure-openai]",
    )

try:
    from llama_index.embeddings.huggingface import (
        HuggingFaceInferenceAPIEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[HuggingFaceInferenceAPIEmbedding.class_name()] = (
        HuggingFaceInferenceAPIEmbedding
    )
except ImportError:
    HuggingFaceInferenceAPIEmbedding = ImportErrorProxy(
        "HuggingFaceInferenceAPIEmbedding",
        "pip install openparse[embeddings-huggingface]",
    )


try:
    from llama_index.embeddings.huggingface_optimum import (
        OptimumEmbedding,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[OptimumEmbedding.class_name()] = OptimumEmbedding
except ImportError:
    OptimumEmbedding = ImportErrorProxy(
        "OptimumEmbedding",
        "pip install openparse[embeddings-huggingface-optimum]",
    )

try:
    from llama_index.embeddings.cohere import CohereEmbedding  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[CohereEmbedding.class_name()] = CohereEmbedding
except ImportError:
    CohereEmbedding = ImportErrorProxy(
        "CohereEmbedding",
        "pip install openparse[embeddings-cohere]",
    )


try:
    from llama_index.embeddings.text_embeddings_inference import (
        TextEmbeddingsInference,
    )  # pants: no-infer-dep

    RECOGNIZED_EMBEDDINGS[TextEmbeddingsInference.class_name()] = (
        TextEmbeddingsInference
    )
except ImportError:
    TextEmbeddingsInference = ImportErrorProxy(
        "TextEmbeddingsInference",
        "pip install openparse[embeddings-text-embeddings-inference]",
    )
