

Processing is how we group related elements together to form a coherent structure. The output are Nodes that represent distinct sections of the document.

## 1. Default Processing 

By default, we use a simple heuristic to group elements together. This works well for many documents.

```python
from openparse import DocumentParser

parser = DocumentParser()
```

## 2. Semantic Processing (Recommended)

Chunking documents is fundamentally about grouping similar semantic nodes together. Perhaps the most powerful way to do this is to use embeddings. **By embedding the text of each node, we can then cluster them together based on their similarity.** 

We currently only support the OpenAI API to generate embeddings.

```python
from openparse import processing, DocumentParser

semantic_pipeline = processing.SemanticIngestionPipeline(
    openai_api_key=OPEN_AI_KEY,
    model="text-embedding-3-large",
    min_tokens=64,
    max_tokens=1024,
)
parser = DocumentParser(
    processing_pipeline=semantic_pipeline,
)
parsed_content = parser.parse(basic_doc_path)
```

#### Notes on Node Size:

We have a bias towards chunking that results in larger nodes - models have increasingly large context windows and we find large nodes to provider bettter context for the model.

A more thorough discussion can be found [here](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5).

