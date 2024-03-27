def output_to_html(headers: list[str], rows: list[list[str]]) -> str:
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


def output_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    markdown_output = "| " + " | ".join(headers) + " |\n"
    markdown_output += "|---" * len(headers) + "|\n"

    for row in rows:
        processed_row = [" " if cell in [None, ""] else cell for cell in row]
        markdown_output += "| " + " | ".join(processed_row) + " |\n"

    return markdown_output
