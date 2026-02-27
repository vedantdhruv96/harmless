"""Tests for harmless.fluid."""

import pytest


class TestFluidDumpBadFile:
    def test_nonexistent_file_raises(self):
        """FluidDump should raise OSError (or subclass) for a missing file."""
        from harmless.fluid import FluidDump

        with pytest.raises(OSError):
            FluidDump("/nonexistent/path/to/dump.h5")


class TestGetDerivedBadKey:
    def test_bad_key_raises_key_error(self):
        """get_derived should raise KeyError for an unrecognised variable name."""
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Build a minimal mock FluidDump without touching HDF5
        from harmless import fluid, diagnostics

        dump = object.__new__(fluid.FluidDump)
        dump.n1 = dump.n2 = dump.n3 = 4
        dump.gam = 4.0 / 3.0
        dump.rho = np.ones((4, 4, 4))
        dump.u = np.ones((4, 4, 4)) * 0.1

        with pytest.raises(KeyError, match="not a recognised diagnostic"):
            dump.get_derived("totally_fake_variable")

    def test_known_key_pg_returns_array(self):
        """get_derived('pg') should return (gam-1)*u without needing a Grid."""
        import numpy as np
        from harmless import fluid

        dump = object.__new__(fluid.FluidDump)
        dump.n1 = dump.n2 = dump.n3 = 4
        dump.gam = 4.0 / 3.0
        dump.u = np.ones((4, 4, 4)) * 3.0

        result = dump.get_derived("pg")
        expected = (4.0 / 3.0 - 1.0) * 3.0
        assert result == pytest.approx(expected)
