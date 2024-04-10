

Processing is how we group related elements together to form a coherent structure. The output are Nodes that represent distinct sections of the document.

<img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/marketing/open-parse-architecture.png">

## 1. Default Processing 

By default, we use a simple heuristic to group elements together. This works well for many documents.

These are mostly just commmon sense transforms - a heading should be grouped with the text that follows it, a bullet list should be grouped together, etc. 

```python
from openparse import DocumentParser

parser = DocumentParser()
```

## 2. Semantic Processing (Recommended)

Chunking documents is fundamentally about grouping similar semantic nodes together. Perhaps the most powerful way to do this is to use embeddings. **By embedding the text of each node, we can then cluster them together based on their similarity.**  

We currently only support the OpenAI API to generate embeddings but plan on adding more options soon.

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

If you're interested in understand how this works, you can see a demo notebook [here](https://github.com/Filimoa/open-parse/blob/main/src/cookbooks/semantic_processing.ipynb).

#### Notes on Node Size:

We have a bias towards chunking that results in larger nodes - models have increasingly large context windows and we find large nodes perform better.

A more thorough discussion can be found [here](https://www.llamaindex.ai/blog/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5).

