# @pytest.mark.parametrize(
#     "criteria, expected_combine",
#     [
#         ("both_small", True),
#         ("either_stub", True),
#     ],
# )
# def test_node_combination_spatially(create_text_element, criteria, expected_combine):
#     node1 = create_text_element(1, 0, 0, 2, 2)
#     node2 = create_text_element(1, 1.5, 1.5, 3.5, 3.5)

#     combined_nodes = _combine_nodes_spatially(
#         [node1, node2], x_error_margin=0, y_error_margin=0, criteria=criteria
#     )

#     if expected_combine:
#         assert len(combined_nodes) == 1, "Nodes should have been combined"
#     else:
#         assert len(combined_nodes) == 2, "Nodes should not have been combined"


# def test_nodes_on_different_pages_should_not_combine(create_text_element):
#     node1 = create_text_element(1, 0, 0, 1, 1)
#     node2 = create_text_element(2, 0, 0, 1, 1)

#     combined_nodes = _combine_nodes_spatially(
#         [node1, node2], x_error_margin=0, y_error_margin=0, criteria="both_small"
#     )

#     assert len(combined_nodes) == 2, "Nodes on different pages should not combine"
