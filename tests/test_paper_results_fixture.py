from tests.paper_checks import verify_paper_results


def test_paper_results_match_fixture():
    counts = verify_paper_results()
    assert counts == {
        "direct_rows": 1,
        "neighbor_rows": 6,
        "runtime_rows": 16,
        "mnist_rows": 20,
        "contamination_cells": 90,
        "smallnorb_rows": 4,
        "smallnorb_primary_pairs": 3,
        "pairwise_rows": 14,
    }
