# Architecture

## Core

#### /PDF

This is really just a wrapper class around a `pdfminer`, `pymupdf` and `pypdf`. We implement some basic visualization / export methods. Would like to migrate away from `pymupdf` for converting pdfs to images due to its licensing.

#### /Schemas

This is where we define the data models for the project. We use pydantic to define these models. This is useful for serialization and validation. Some methods in these classes need more robust testing asap. In general we prefer freezing as many attributes as possible to avoid unexpected behavior.

## Elements 

#### /Text

This module implements basic text parsing along with basic markdown support.

We parse text into markdown by looking at the font size and style charachter by charachter. This gets combined into a span which represents a string of charachters with the same styling. 

Spans get combined into lines and lines get combined into elements. Elements are the basic building blocks of the document. They can be headings, paragraphs, lists of bullets, etc.

Optionally we can use PyMuPDF to OCR the document. This is not recommended as a default due to the additional computational cost and inherent inaccuracies of OCR. We're looking at integrating [doctr](https://github.com/mindee/doctr).  Early tests make this library seem very heavy and slow.

Here's an article that goes into more details on available [OCR libraries](https://source.opennews.org/articles/our-search-best-ocr-tool-2023/).

#### /Tables

We implement a few different methods for table extraction.

<a href="https://huggingface.co/microsoft/table-transformer-detection" class="external-link" target="_blank">Table Transformer</a> can be used for both detecting tables and their contents.  

PyMuPDF is also can be used to table detection and content extraction.

Lastly unitable is our recommended approach for table extraction. It is a transformers based approach with **state-of-the-art** performance. Unfortunately, its performance is hindered by the fact that we still need to use the table-transformers model to detect table bounding boxes. Table Transformers's performance leaves a lot to be desired and may miss some tables or crop them incorrectly. **If you're aware of a stronger perfoming model, please let us know.**

We're also looking at speeding unitable up. This can either be done by quantizing the model or by using the smaller, 70M parameter model they released. Unfortunately, the smaller model was not fine tuned so this is holding us back from implementing it. You can see the published paper [here](https://arxiv.org/abs/2403.04822).

A ton of credit goes to the unitable team - they've done an amazing job making their research reproducible. You can find the original repository with full training code [here](https://github.com/poloclub/unitable).

## Processing Pipeline

#### /Processing

This is mostly a collection of fast heuristics to combine and split elements. The main idea is to have a fast and simple way to process the data. We looked into more complex methods like [Layout Parser Documentation](https://layout-parser.github.io/) but did not find the result compelling enough to full integrate. 

We also have a semantic processing pipeline that uses embeddings to cluster similar nodes together. This is powerful but we need to look into more robust ways to cluster these since we currently hardcode similarity thresholds.

You can implement you own rules by defining a subclass of `ProcessingStep` and adding a `process` method. This method should take a list of nodes and return a list of nodes. 

Then you can add this to the pipeline by calling `add_step` on the `DocumentParser` object or create a new pipeline object with your custom steps.

This can be done by subclassing `ProcessingPipeline` and adding your custom steps to the `transformations` attribute.
