from typing import List, Tuple


def output_to_html(headers: List[str], rows: List[List[str]]) -> str:
    html_output = '<table border="1">\n'

    html_output += "<tr>"

    for header in headers:
        html_output += f"<th>{header}</th>"
    html_output += "</tr>\n"

    for row in rows:
        html_output += "<tr>"
        for cell in row:
            html_output += f'<td>{cell or " "}</td>'
        html_output += "</tr>\n"

    html_output += "</table>"

    return html_output


def output_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    markdown_output = ""
    if headers is not None:
        for header in headers:
            safe_header = "" if header is None else header
            markdown_output += "| " + safe_header + " "

    markdown_output += "|\n"
    markdown_output += "|---" * len(headers) + "|\n"

    for row in rows:
        processed_row = [
            " " if cell in [None, ""] else cell.replace("\n", " ") for cell in row
        ]
        markdown_output += "| " + " | ".join(processed_row) + " |\n"

    return markdown_output


def combine_header_and_table_bboxes(
    bbox1: Tuple[float, float, float, float], bbox2: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    x0 = min(bbox1[0], bbox2[0])
    y0 = min(bbox1[1], bbox2[1])
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[3], bbox2[3])

    return x0, y0, x1, y1
