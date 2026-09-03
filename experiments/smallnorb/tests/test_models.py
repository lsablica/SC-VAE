from __future__ import annotations

import torch

from experiments.smallnorb.config import (
    ALL_FAMILIES,
    RunConfig,
)
from experiments.smallnorb.models import build_model


def _config(family: str) -> RunConfig:
    return RunConfig(
        family=family,
        seed=0,
        epochs=1,
        num_workers=0,
        mixed_precision=False,
    )


def test_every_family_uses_identical_backbone_decoder_and_33d_input():
    summaries = {}
    inputs = torch.rand(2, 1, 64, 64)
    for family in ALL_FAMILIES:
        model = build_model(_config(family), torch.device("cpu"))
        output = model(inputs)
        assert output.reconstruction.shape == inputs.shape
        assert output.latent.shape == (2, 33)
        assert output.kl.shape == (2,)
        summaries[family] = model.parameter_summary()
    assert len(
        {value["shared_encoder"] for value in summaries.values()}
    ) == 1
    assert len(
        {value["shared_decoder"] for value in summaries.values()}
    ) == 1


def test_gaussian_decoder_padding_is_constant_zero():
    for family in ("gaussian_isotropic", "gaussian_diagonal"):
        model = build_model(_config(family), torch.device("cpu"))
        parameters = model.encode(torch.rand(3, 1, 64, 64))
        representative = model.posterior.representative(parameters)
        sample = model.posterior.sample(parameters)
        assert torch.equal(
            representative[:, -1], torch.zeros(3)
        )
        assert torch.equal(sample[:, -1], torch.zeros(3))


def test_no_encoder_decoder_skip_path():
    model = build_model(_config("spcauchy"), torch.device("cpu"))
    assert not any(
        "skip" in name.lower() for name, _ in model.named_modules()
    )


def test_deep_architecture_is_shared_and_strictly_larger():
    default = build_model(
        _config("spcauchy"), torch.device("cpu")
    ).parameter_summary()
    summaries = []
    for family in ("spcauchy", "gaussian_isotropic"):
        config = RunConfig(
            family=family,
            seed=0,
            architecture="deep_residual_cnn",
            epochs=1,
            num_workers=0,
            mixed_precision=False,
        )
        model = build_model(config, torch.device("cpu"))
        output = model(torch.rand(2, 1, 64, 64))
        assert output.latent.shape == (2, 33)
        summaries.append(model.parameter_summary())
    assert (
        summaries[0]["shared_encoder"]
        == summaries[1]["shared_encoder"]
        > default["shared_encoder"]
    )
    assert (
        summaries[0]["shared_decoder"]
        == summaries[1]["shared_decoder"]
        > default["shared_decoder"]
    )
