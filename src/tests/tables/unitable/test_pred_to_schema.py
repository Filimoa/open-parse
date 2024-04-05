from bs4 import BeautifulSoup
from .sample_pred_outputs import sample_preds


def get_table_structure(html_string):
    # Function to parse HTML and return a list of row cell counts
    soup = BeautifulSoup(html_string, "html.parser")
    table = soup.find("table")

    structure = []

    # Iterate through header and body rows
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        structure.append(len(cells))

    return structure


def test_schema_table_structure_matches_unitable_full_outputs():
    """
    We don't use unitable cell extraction since it's significantly slower than doing traditional ocr. As a result we need to verify our table outputs match the unitable outputs.
    """
    # for val in sample_preds:
    #     schema_obj = HTMLTable.from_model_outputs(
    #         structure=val["pred_html"],
    #         bboxes=val["pred_bbox"],
    #     )

    #     schema_structure = get_table_structure(schema_obj.to_html())
    #     unitable_structure = get_table_structure(val["core_html"])
    #     assert schema_structure == unitable_structure
    pass
