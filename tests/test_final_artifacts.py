from tests.paper_checks import verify_final_artifacts


def test_compact_final_artifact_manifests() -> None:
    counts = verify_final_artifacts()
    assert counts == {
        "latent_layer": 23,
        "mnist": 15,
        "contaminated_directional": 10,
        "smallnorb": 17,
    }
