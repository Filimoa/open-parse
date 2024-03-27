from openparse.tables.table_transformers.schemas import _TableCellModelOutput

# from src.tables.schemas import _TableCellModelOutput


# evals/data/tables/naic-numerical-list-of-companies-page-94.pdf
sample_get_table_content_output = [
    _TableCellModelOutput(
        label="table row",
        confidence=0.9939164519309998,
        bbox=(
            35.288272164084674,
            408.22346635298294,
            690.0443794944069,
            424.6131758256392,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9996691942214966,
        bbox=(
            35.15542533180928,
            490.4657454057173,
            690.3566963889382,
            506.47007890181106,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.99506676197052,
        bbox=(
            35.129248879172565,
            160.296967939897,
            690.2205879905007,
            176.68047471479935,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9961244463920593,
        bbox=(
            35.17559745094991,
            375.1639875932173,
            690.0448677756569,
            391.70527787642044,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9988767504692078,
        bbox=(
            452.4295723655007,
            105.89149613813919,
            497.28964926979756,
            519.6490339799361,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9969657063484192,
        bbox=(
            360.2500069358132,
            105.74062486128372,
            410.5702278830788,
            519.9760298295455,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9950108528137207,
        bbox=(
            35.18905188820577,
            391.7489180131392,
            690.0088570334694,
            408.2126936479048,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9994648098945618,
        bbox=(
            411.57538535378194,
            105.81761880354446,
            452.10687949440694,
            519.731522993608,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9994288086891174,
        bbox=(
            35.216128609397174,
            325.67996354536575,
            690.3316109397194,
            342.2242598100142,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9953063130378723,
        bbox=(
            35.20100333473897,
            143.6793379350142,
            690.1937935569069,
            160.27500291304153,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9981030225753784,
        bbox=(
            35.22977759621358,
            123.52590699629349,
            690.1675484397194,
            143.69812150435013,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9999344348907471,
        bbox=(
            35.117327950217486,
            105.96672578291458,
            206.06037070534444,
            519.4698347611861,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9996514320373535,
        bbox=(
            34.91754843971944,
            209.7849287553267,
            690.2598946311257,
            226.38141007856888,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9987955093383789,
        bbox=(
            35.13444068215108,
            276.3870100541548,
            690.0095894553444,
            292.8090986772017,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9938255548477173,
        bbox=(
            35.265181801535846,
            424.74266190962356,
            690.0817940451882,
            441.19603105024856,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9993770718574524,
        bbox=(
            35.005008003928424,
            226.48010392622513,
            690.1912300803444,
            242.9812483354048,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9996106028556824,
        bbox=(
            34.99921347878194,
            176.78753800825638,
            690.1875069358132,
            193.39771409468216,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.999058187007904,
        bbox=(
            35.12911917946553,
            474.03270097212356,
            690.1881783225319,
            490.56614823774856,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.999068558216095,
        bbox=(
            35.018870613791705,
            243.06179948286575,
            690.3463204123757,
            259.6156630082564,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9995618462562561,
        bbox=(
            35.03798987648702,
            193.34961076216263,
            690.2454903342507,
            209.94493241743606,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9995167255401611,
        bbox=(
            35.13354041359639,
            292.3361525102095,
            690.2454292990944,
            308.8550428910689,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9995043277740479,
        bbox=(
            35.23306205055928,
            105.57804055647415,
            690.1783516623757,
            123.37710710005325,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9793805480003357,
        bbox=(
            496.0791084983132,
            105.98925919966263,
            521.5678169944069,
            519.3316511674361,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9991280436515808,
        bbox=(
            35.10216834328389,
            341.9986738725142,
            690.5628731467507,
            358.4926008744673,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9999399185180664,
        bbox=(
            521.5474922873757,
            105.70714898542923,
            690.0865547873757,
            519.495927290483,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9976800084114075,
        bbox=(
            35.068866036154986,
            506.24861283735794,
            690.4476998069069,
            519.2454695268111,
        ),
    ),
    _TableCellModelOutput(
        label="table column header",
        confidence=0.9989497065544128,
        bbox=(
            35.182784340598346,
            105.63359590010208,
            690.2292549826882,
            123.22541184858841,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9968582391738892,
        bbox=(
            35.039038918235065,
            358.5333418412642,
            690.1406319358132,
            375.10054154829544,
        ),
    ),
    _TableCellModelOutput(
        label="table column",
        confidence=0.9961622953414917,
        bbox=(
            206.15311362526631,
            105.64504189924759,
            361.92600180886006,
            519.362382368608,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9984205961227417,
        bbox=(
            35.26417472145772,
            457.65477128462356,
            690.0900337912819,
            474.10359330610794,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9983760118484497,
        bbox=(
            34.888621590354205,
            259.36580033735794,
            690.3048165061257,
            275.8905958695845,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.9978849291801453,
        bbox=(
            35.130687020041705,
            441.1501936479048,
            690.1639473655007,
            457.64375443892044,
        ),
    ),
    _TableCellModelOutput(
        label="table spanning cell",
        confidence=0.9597509503364563,
        bbox=(
            32.47110869667745,
            130.52832932905716,
            209.55084922096944,
            516.4211592240767,
        ),
    ),
    _TableCellModelOutput(
        label="table row",
        confidence=0.99912029504776,
        bbox=(
            35.06736304543233,
            308.5428633256392,
            690.4191963889382,
            325.1110243363814,
        ),
    ),
    _TableCellModelOutput(
        label="table",
        confidence=0.9999959468841553,
        bbox=(
            35.22020652077413,
            105.64571900801224,
            690.1153023459694,
            519.1645063920455,
        ),
    ),
]


# def test_table_from_model_outputs():
#     image_size = (792, 612)
#     page_size = (792.0, 612.0)
#     table_bbox = (
#         56.02,
#         180.17,
#         702.35,
#         460.68,
#     )
#     table_cells = sample_get_table_content_output

#     headers = [
#         cell
#         for cell in table_cells
#         if cell.is_header and cell.confidence > ml.MIN_CELL_CONFIDENCE
#     ]
#     rows = [
#         cell
#         for cell in table_cells
#         if cell.is_row and cell.confidence > ml.MIN_CELL_CONFIDENCE
#     ]
#     cols = [
#         cell
#         for cell in table_cells
#         if cell.is_column and cell.confidence > ml.MIN_CELL_CONFIDENCE
#     ]

#     assert len(headers) == 1
#     header_objs = ml._preprocess_header_cells(headers, cols, image_size, page_size)
#     assert len(header_objs) == 1

#     assert len(rows) == 26  # 24 rows + 1 spanning cell
#     row_objs = ml._process_row_cells(rows, cols, header_objs, image_size, page_size)
#     # row_objs = ml._drop_duplicates(row_objs, threshold=0.3)
#     assert len(row_objs) == 25  # 24 rows + 1 spanning cell
