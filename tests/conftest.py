"""Shared pytest fixtures for harmless tests."""

import numpy as np
import pytest
from harmless.grid import Grid


@pytest.fixture(scope="module")
def small_cartesian_grid():
    """A tiny 4x4x4 Cartesian Grid for fast testing."""
    return Grid(
        "cartesian",
        4,
        4,
        4,
        a=0.0,
        r_out=100.0,
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
    )


@pytest.fixture(scope="module")
def small_eks_grid():
    """A small EKS Grid for fast testing.
    n1=8 is required: the HARM EH zone formula needs n1 > 5.5 to produce a valid r_in.
    """
    return Grid(
        "eks",
        8,
        4,
        4,
        a=0.9375,
        r_out=1000.0,
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
    )


@pytest.fixture(scope="module")
def small_mks_grid():
    """A small MKS Grid for fast testing. n1=8 for the same reason as small_eks_grid."""
    return Grid(
        "mks",
        8,
        4,
        4,
        a=0.9375,
        r_out=1000.0,
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
    )


@pytest.fixture(scope="module")
def small_fmks_grid():
    """A small FMKS Grid for fast testing.
    n1=8 for the same reason as small_eks_grid.
    """
    return Grid(
        "fmks",
        8,
        4,
        4,
        a=0.9375,
        r_out=1000.0,
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
    )
