"""Tests for harmless.grid."""

import numpy as np
import pytest
from harmless.grid import Grid, lower_vec, raise_vec, dot_vec, inv_scalar


class TestGridCartesian:
    def test_construction(self, small_cartesian_grid):
        G = small_cartesian_grid
        assert G.coord_sys == "cartesian"
        assert G.n1 == G.n2 == G.n3 == 4

    def test_coordinate_shapes(self, small_cartesian_grid):
        G = small_cartesian_grid
        assert G.x1.shape == (4, 4, 4)
        assert G.x2.shape == (4, 4, 4)
        assert G.x3.shape == (4, 4, 4)

    def test_metric_shapes(self, small_cartesian_grid):
        G = small_cartesian_grid
        assert G.gcov.shape == (4, 4, 4, 4, 4)
        assert G.gcon.shape == (4, 4, 4, 4, 4)
        # gdet is set to scalar 1. in cartesian() — just check value, not shape
        assert G.gdet == pytest.approx(1.0)
        assert G.lapse.shape == (4, 4, 4)

    def test_cartesian_metric_is_minkowski(self, small_cartesian_grid):
        G = small_cartesian_grid
        # Diagonal should be (-1, 1, 1, 1) everywhere
        assert G.gcov[0, 0, 0, 0, 0] == pytest.approx(-1.0)
        assert G.gcov[0, 0, 0, 1, 1] == pytest.approx(1.0)
        assert G.gcov[0, 0, 0, 2, 2] == pytest.approx(1.0)
        assert G.gcov[0, 0, 0, 3, 3] == pytest.approx(1.0)

    def test_lapse_is_one_for_cartesian(self, small_cartesian_grid):
        G = small_cartesian_grid
        assert np.all(G.lapse == pytest.approx(1.0))


class TestGridEKS:
    def test_construction(self, small_eks_grid):
        G = small_eks_grid
        assert G.coord_sys == "eks"

    def test_metric_shapes(self, small_eks_grid):
        G = small_eks_grid
        # Fixture is 8x4x4
        assert G.gcov.shape == (8, 4, 4, 4, 4)
        assert G.gcon.shape == (8, 4, 4, 4, 4)
        assert G.gdet.shape == (8, 4, 4)

    def test_spherical_coordinate_shapes(self, small_eks_grid):
        G = small_eks_grid
        assert G.r.shape == (8, 4, 4)
        assert G.th.shape == (8, 4, 4)
        assert G.phi.shape == (8, 4, 4)

    def test_r_is_positive(self, small_eks_grid):
        G = small_eks_grid
        assert np.all(G.r > 0.0)

    def test_theta_in_range(self, small_eks_grid):
        G = small_eks_grid
        assert np.all(G.th >= 0.0) and np.all(G.th <= np.pi)

    def test_gdet_positive(self, small_eks_grid):
        G = small_eks_grid
        assert np.all(G.gdet > 0.0)


class TestGridMKS:
    def test_construction(self, small_mks_grid):
        G = small_mks_grid
        assert G.coord_sys == "mks"

    def test_gdet_positive(self, small_mks_grid):
        G = small_mks_grid
        assert np.all(G.gdet > 0.0)


class TestGridFMKS:
    def test_construction(self, small_fmks_grid):
        G = small_fmks_grid
        assert G.coord_sys == "fmks"

    def test_gdet_positive(self, small_fmks_grid):
        G = small_fmks_grid
        assert np.all(G.gdet > 0.0)


class TestGridInvalidCoordSys:
    def test_invalid_coord_sys_exits(self):
        with pytest.raises(SystemExit):
            Grid(
                "bogus",
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


class TestGridHelpers:
    def test_lower_vec_shape(self, small_cartesian_grid):
        G = small_cartesian_grid
        vcon = np.random.rand(4, 4, 4, 4)
        vcov = lower_vec(vcon, G)
        assert vcov.shape == (4, 4, 4, 4)

    def test_raise_vec_shape(self, small_cartesian_grid):
        G = small_cartesian_grid
        vcov = np.random.rand(4, 4, 4, 4)
        vcon = raise_vec(vcov, G)
        assert vcon.shape == (4, 4, 4, 4)

    def test_lower_then_raise_is_identity(self, small_cartesian_grid):
        # Use Cartesian (Minkowski) grid: gcon = gcov = diag(-1,1,1,1), trivially exact inverse
        G = small_cartesian_grid
        rng = np.random.default_rng(42)
        vcon = rng.random((4, 4, 4, 4))
        recovered = raise_vec(lower_vec(vcon, G), G)
        assert recovered == pytest.approx(vcon, abs=1e-12)

    def test_dot_vec_shape(self, small_cartesian_grid):
        G = small_cartesian_grid
        vcon = np.random.rand(4, 4, 4, 4)
        vcov = lower_vec(vcon, G)
        result = dot_vec(vcov, vcon)
        assert result.shape == (4, 4, 4)

    def test_inv_scalar(self):
        x = np.array([1.0, 2.0, 4.0])
        result = inv_scalar(x)
        assert result == pytest.approx([1.0, 0.5, 0.25])
