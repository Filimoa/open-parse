import re

import openparse


def test_parse_doc():
    basic_doc_path = "src/evals/data/full-pdfs/mock-1-page-lease.pdf"
    parser = openparse.DocumentParser()
    parsed_basic_doc = parser.parse(basic_doc_path)
    assert len(parsed_basic_doc.nodes) >= 1
    assert parsed_basic_doc.nodes[0].text.startswith("**MOCK LEASE AGREEMENT**")


def get_cols(html_string):
    pattern = r"<thead>(.*?)</thead>"
    match = re.search(pattern, html_string, re.DOTALL)
    if match:
        return match.group(1)
    return None


def test_parse_tables_with_table_transformers():
    doc_with_tables_path = (
        "src/evals/data/tables/naic-numerical-list-of-companies-page-94.pdf"
    )

    parser = openparse.DocumentParser(
        table_args={"parsing_algorithm": "table-transformers"}
    )
    parsed_doc2 = parser.parse(doc_with_tables_path)
    assert len(parsed_doc2.nodes) >= 1
    found_text = get_cols(parsed_doc2.nodes[0].text)

    assert found_text is not None
    assert "GROUP NAME" in found_text
    assert "GROUP" in found_text
    assert "CO NO" in found_text
    assert "STMT" in found_text
    assert "STATUS" in found_text
    assert "ST" in found_text
    assert "COMPANY NAME" in found_text


def test_parse_tables_with_pymupdf():
    doc_with_tables_path = "src/evals/data/tables/meta-2022-10k-page-69.pdf"

    parser = openparse.DocumentParser(table_args={"parsing_algorithm": "pymupdf"})

    parsed_doc2 = parser.parse(doc_with_tables_path)
    assert len(parsed_doc2.nodes) >= 1
    assert parsed_doc2.nodes[-1].text.startswith("<table")


def test_to_llama_index_nodes():
    basic_doc_path = "src/evals/data/full-pdfs/mock-1-page-lease.pdf"
    parser = openparse.DocumentParser()
    parsed_basic_doc = parser.parse(basic_doc_path)

    nodes = parsed_basic_doc.to_llama_index_nodes()
    assert len(nodes) >= 1
