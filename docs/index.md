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

```python
import openparse

basic_doc_path = "./sample-docs/mobile-home-manual.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)

for node in parsed_basic_doc.nodes:
    print(node)
```

<br>

**📓 Try the sample notebook** <a href="https://colab.research.google.com/drive/1Z5B5gsnmhFKEFL-5yYIcoox7-jQao8Ep?usp=sharing" class="external-link" target="_blank">here</a>

<br><br>
