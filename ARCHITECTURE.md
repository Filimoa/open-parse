# Library Motivation:

**Open source:**  Most libraries rely on chunking documents based on fixed string lengths. This naive approach throws out an enormous amount of valuable information that's hidden in the document structure. It also makes it challenging to display query citations down the line.  We find quality citations to be invaluable in any non trivial RAG application.

**Commercial Offerings:** We've found these to either be cost prohibitive or lacking in performance.  Foundational model providers like OpenAI and Google have also started implementing their own files API but these are black boxes and don't support complex querying.



# Architecture

## Element Extraction

#### 1. /Text

This module implements basic text parsing along with basic markdown support.

#### 2. /Tables

We implement <a href="https://huggingface.co/microsoft/table-transformer-detection" class="external-link" target="_blank">Table Transformer</a> for parsing tables and their contents. Tables can be exported in a couple different formats - str, markdown or html. 

## Processing Pipeline

#### 3. /Processing

This is mostly a collection of fast heuristics to combine and split elements. The main idea is to have a fast and simple way to process the data. We looked into more complex methods like [Layout Parser Documentation](https://layout-parser.github.io/) but did not find the result compelling enough to full integrate. 

You can implement you own rules by defining a subclass of `ProcessingStep` and adding it to the processing planner.

#### 4. /Post Processing

There's promising techniques that require an LLM to integrate - we implement these here.  We currently only support OpenAI.

- Using embeddings to combine similar nodes. This is especially useful to combine nodes that stretch across pages.
- Use an LLM to describe the contents of an image or graph. This helps with recall.
