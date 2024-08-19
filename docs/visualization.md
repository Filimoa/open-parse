## Displaying Results within a Jupyter Notebook

The `Node` class has built in support for rendering it's text contents as markdown in a `jupyter` notebook.

```py
import openparse

basic_doc_path = "./sample-docs/mobile-home-manual.pdf"
parser = openparse.DocumentParser()
parsed_basic_doc = parser.parse(basic_doc_path)

for node in parsed_basic_doc.nodes:
    display(node)
```
<br/>
<p align="center">
    <img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/marketing/pretty-markdown-nodes.webp" width="650" />
</p>
You can also display the results directly overlayed on the original pdf.

```python
pdf = openparse.Pdf(basic_doc_path)
pdf.display_with_bboxes(
    parsed_basic_doc.nodes,
)
```

<br/>
<p align="center">
    <img src="https://sergey-filimonov.nyc3.digitaloceanspaces.com/open-parse/docs/marked-up-outputs.jpeg" width="450" />
</p>

## Exporting Overlayed Results to a Pdf

You can also export the results marked up over the original pdf to a seperate pdf file.

```python
pdf = openparse.Pdf(basic_doc_path)
pdf.export_with_bboxes(
    parsed_basic_doc.nodes,
    output_pdf="./sample-docs/mobile-home-manual-marked-up.pdf"
)
```
