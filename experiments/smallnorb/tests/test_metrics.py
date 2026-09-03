from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.smallnorb.metrics import (
    circular_absolute_error_degrees,
    reconstruction_metric_vectors,
    slerp,
    structural_similarity,
)


def test_identical_images_have_zero_error_infinite_limit_psnr_and_unit_ssim():
    images = torch.rand(3, 1, 64, 64)
    metrics = reconstruction_metric_vectors(images, images, 0.2)
    assert torch.equal(metrics["pixel_mse"], torch.zeros(3))
    assert torch.equal(metrics["reconstruction_nll"], torch.zeros(3))
    assert torch.all(metrics["psnr_db"] > 60)
    assert torch.allclose(metrics["ssim"], torch.ones(3), atol=1e-6)


def test_reconstruction_nll_matches_fixed_variance_definition():
    images = torch.zeros(2, 1, 64, 64)
    reconstructions = torch.ones_like(images)
    metrics = reconstruction_metric_vectors(
        images, reconstructions, sigma_x=0.2, include_ssim=False
    )
    assert torch.allclose(
        metrics["reconstruction_nll"],
        torch.full((2,), 4096 / (2 * 0.2**2)),
    )


def test_ssim_decreases_for_unrelated_image():
    image = torch.rand(1, 1, 64, 64)
    unrelated = 1.0 - image
    assert float(structural_similarity(image, image)) == pytest.approx(
        1.0, abs=1e-6
    )
    assert float(structural_similarity(image, unrelated)) < 0.5


def test_circular_error_wraps_at_seam():
    errors = circular_absolute_error_degrees(
        np.array([350, 10, 90]), np.array([10, 350, 270])
    )
    assert np.allclose(errors, [20, 20, 180])


def test_slerp_stays_on_unit_sphere_and_hits_endpoints():
    start = torch.tensor([[1.0, 0.0, 0.0]])
    end = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.cat(
        [slerp(start, end, fraction) for fraction in (0.0, 0.5, 1.0)]
    )
    assert torch.allclose(values.norm(dim=-1), torch.ones(3))
    assert torch.allclose(values[0], start[0])
    assert torch.allclose(values[-1], end[0])
