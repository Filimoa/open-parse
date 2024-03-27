from openparse.tables.pymupdf.parse import output_to_html, output_to_markdown


def test_parse_output_to_markdown():
    # Standard case
    headers = ["Year", "Revenue", "Expenses"]
    rows = [["2022", "100,000", "50,000"], ["2021", "90,000", "45,000"]]
    expected_output = (
        "| Year | Revenue | Expenses |\n"
        "|---|---|---|\n"
        "| 2022 | 100,000 | 50,000 |\n"
        "| 2021 | 90,000 | 45,000 |\n"
    )
    assert output_to_markdown(headers, rows) == expected_output, "Standard case failed"

    # Case with missing values
    headers = ["Year", "Revenue", "Expenses"]
    rows = [
        ["2022", "100,000", None],  # Missing value represented with extra space
        ["2021", "", "45,000"],  # Empty string represented with extra space
    ]
    expected_output = (
        "| Year | Revenue | Expenses |\n"
        "|---|---|---|\n"
        "| 2022 | 100,000 |   |\n"  # Notice the extra spaces before closing |
        "| 2021 |   | 45,000 |\n"
    )
    actual_output = output_to_markdown(headers, rows)
    assert actual_output == expected_output, "Case with missing values failed"

    # Edge case with no data
    headers = ["Year", "Revenue"]
    rows = []
    expected_output = "| Year | Revenue |\n" "|---|---|\n"
    assert (
        output_to_markdown(headers, rows) == expected_output
    ), "Edge case with no data failed"


def test_output_to_html():
    # Standard case
    headers = ["Year", "Revenue", "Expenses"]
    rows = [["2022", "100,000", "50,000"], ["2021", "90,000", "45,000"]]
    expected_output = (
        '<table border="1">\n'
        "<tr><th>Year</th><th>Revenue</th><th>Expenses</th></tr>\n"
        "<tr><td>2022</td><td>100,000</td><td>50,000</td></tr>\n"
        "<tr><td>2021</td><td>90,000</td><td>45,000</td></tr>\n"
        "</table>"
    )
    assert output_to_html(headers, rows) == expected_output, "Standard case failed"

    # Case with missing values
    headers = ["Year", "Revenue", "Expenses"]
    rows = [
        ["2022", "100,000", None],
        ["2021", "", "45,000"],
    ]
    expected_output = (
        '<table border="1">\n'
        "<tr><th>Year</th><th>Revenue</th><th>Expenses</th></tr>\n"
        "<tr><td>2022</td><td>100,000</td><td> </td></tr>\n"
        "<tr><td>2021</td><td> </td><td>45,000</td></tr>\n"
        "</table>"
    )
    actual_output = output_to_html(headers, rows)
    assert actual_output == expected_output, "Case with missing values failed"

    # Edge case with no data
    headers = ["Year", "Revenue"]
    rows = []
    expected_output = (
        '<table border="1">\n' "<tr><th>Year</th><th>Revenue</th></tr>\n" "</table>"
    )
    assert (
        output_to_html(headers, rows) == expected_output
    ), "Edge case with no data failed"
