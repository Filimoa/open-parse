# Welcome to Open Parse

**Easily chunk complex documents the same way a human would.**  

Chunking documents is a challenging task that underpins any RAG system.  High quality results are critical to a sucessful AI application, yet most open-source libraries are limited in their ability to handle complex documents.  

Open Parse is designed to fill this gap by providing a flexible, easy-to-use library capable of visually discerning document layouts and chunking them effectively.

## Features

- 🔍 Visually-Driven: Open-Parse visually analyzes documents for superior LLM input, going beyond naive text splitting.
- ✍️ Markdown Support: Basic markdown support for parsing headings, bold and italics.
- 📊 High-Precision Table Support: Extract tables into clean Markdown formats with accuracy that surpasses traditional tools.
- 🛠️ Extensible: Easily implement your own post-processing steps.
- 💡Intuitive: Great editor support. Completion everywhere. Less time debugging.

<br><br>
![Transformation](https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/docs/transformation.webp)


## Quick Start

## Basic Example

```python
import openparse

basic_doc_path = "./sample-docs/mobile-home-manual.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)

for node in parsed_basic_doc.nodes:
    print(node)
```

**📓 Try the sample notebook** <a href="https://colab.research.google.com/drive/1Z5B5gsnmhFKEFL-5yYIcoox7-jQao8Ep?usp=sharing" class="external-link" target="_blank">here</a>


## Semantic Processing Example

Chunking documents is fundamentally about grouping similar semantic nodes together. By embedding the text of each node, we can then cluster them together based on their similarity.

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

**📓 Sample notebook** <a href="https://github.com/Filimoa/open-parse/blob/main/src/cookbooks/semantic_processing.ipynb" class="external-link" target="_blank">here</a>

<br>



## Cookbooks

https://github.com/Filimoa/open-parse/tree/main/src/cookbooks


## Sponsors

<!-- sponsors -->

<a href="https://www.data.threesigma.ai/filings-ai" target="_blank" title="Three Sigma: AI for insurance filings."><img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/marketing/three-sigma-wide.png" width="250"></a>

<!-- /sponsors -->

Does your use case need something special? Reach [out](https://www.linkedin.com/in/sergey-osu/).
