import sys
import h5py
from harmless.io.base import BaseDump

__all__ = ["KHARMADump"]


class KHARMADump(BaseDump):
    """Reader for KHARMA HDF5 fluid dump files (.phdf5 / .h5).

    Stores primitives and header metadata. Derived quantities are
    computed on demand via :meth:`get_derived`.
    """

    def __init__(self, fname, extras=None):
        """Constructor method for KHARMADump.

        :param fname: Path to the KHARMA dump file
        :type fname: str

        :param extras: List of extra quantity keys to read (divB, fail,
            fixup, imex_errors, solver_failures), defaults to None
        :type extras: list, optional
        """
        dfile = h5py.File(fname, "r")
        self.t = dfile["t"][()]
        self.gam = dfile["header/gam"][()]

        self.n1 = dfile["header/n1"][()]
        self.n2 = dfile["header/n2"][()]
        self.n3 = dfile["header/n3"][()]
        self.coord_sys = dfile["header/metric"][()].decode("UTF-8")
        self.a = dfile["header/geom/" + self.coord_sys + "/a"][()]
        self.startx1 = dfile["header/geom/startx1"][()]
        self.startx2 = dfile["header/geom/startx2"][()]
        self.startx3 = dfile["header/geom/startx3"][()]
        self.r_out = dfile["header/geom/r_out"][()]

        if self.coord_sys == "mks":
            self.hslope = dfile["header/geom/mks/hslop"][()]

        if (self.coord_sys == "mmks") or (self.coord_sys == "fmks"):
            self.hslope = dfile["header/geom/mmks/hslope"][()]
            self.mks_smooth = dfile["header/geom/mmks/mks_smooth"][()]
            self.poly_alpha = dfile["header/geom/mmks/poly_alpha"][()]
            self.poly_xt = dfile["header/geom/mmks/poly_xt"][()]

        self.has_electrons = dfile["header/has_electrons"][()]

        if "header/imex" in dfile:
            self.jacobian_eps = dfile["header/imex/jacobian_eps"][()]
            self.linesearch_eps = dfile["header/imex/linesearch_eps"][()]
            self.rootfinder_tolerance = dfile["header/imex/rootfinder_tolerance"][()]
            self.max_nonlinear_iterations = dfile[
                "header/imex/max_nonlinear_iterations"
            ][()]
            self.max_linesearch_iterations = dfile[
                "header/imex/max_linesearch_iterations"
            ][()]

            if "header/imex/emhd" in dfile:
                self.conduction = dfile["header/imex/emhd/conduction"][()]
                self.viscosity = dfile["header/imex/emhd/viscosity"][()]
                self.conduction_alpha = dfile["header/imex/emhd/conduction_alpha"][()]
                self.viscosity_alpha = dfile["header/imex/emhd/viscosity_alpha"][()]
                self.ho_conduction = dfile[
                    "header/imex/emhd/higher_order_terms_conduction"
                ][()]
                self.ho_viscosity = dfile[
                    "header/imex/emhd/higher_order_terms_viscosity"
                ][()]
                self.tau = dfile["header/imex/emhd/tau"][()]

        self.prim_names = list(
            p.decode("UTF-8") for p in dfile["header/prim_names"][()]
        )
        self.rho = dfile["prims"][()][Ellipsis, 0]
        self.u = dfile["prims"][()][Ellipsis, 1]
        self.u1 = dfile["prims"][()][Ellipsis, 2]
        self.u2 = dfile["prims"][()][Ellipsis, 3]
        self.u3 = dfile["prims"][()][Ellipsis, 4]
        if "header/imex/emhd" in dfile:
            if self.conduction:
                self.qtilde = dfile["prims"][()][Ellipsis, 5]
                if self.viscosity:
                    self.dPtilde = dfile["prims"][()][Ellipsis, 6]
                    self.B1 = dfile["prims"][()][Ellipsis, 7]
                    self.B2 = dfile["prims"][()][Ellipsis, 8]
                    self.B3 = dfile["prims"][()][Ellipsis, 9]
                else:
                    self.B1 = dfile["prims"][()][Ellipsis, 6]
                    self.B2 = dfile["prims"][()][Ellipsis, 7]
                    self.B3 = dfile["prims"][()][Ellipsis, 8]
            elif self.viscosity:
                self.dPtilde = dfile["prims"][()][Ellipsis, 5]
                self.B1 = dfile["prims"][()][Ellipsis, 6]
                self.B2 = dfile["prims"][()][Ellipsis, 7]
                self.B3 = dfile["prims"][()][Ellipsis, 8]
        else:
            self.B1 = dfile["prims"][()][Ellipsis, 5]
            self.B2 = dfile["prims"][()][Ellipsis, 6]
            self.B3 = dfile["prims"][()][Ellipsis, 7]

        if extras is not None:
            self.get_extras(dfile, extras)

        dfile.close()

        self.cache = {}
        self.G = None

    def get_extras(self, dfile, extras):
        """Read additional diagnostic quantities from the dump file.

        Valid keys: ``divB``, ``fail``, ``fixup``, ``imex_errors``,
        ``solver_failures``.

        :param dfile: Open HDF5 file object (passed by constructor)
        :type dfile: h5py.File

        :param extras: List of extra quantity keys to read
        :type extras: list
        """
        for var in extras:
            if var == "divB":
                self.divB = dfile["extras/divB"][()]
            elif var == "fail":
                self.fail = dfile["extras/fail"][()]
            elif var == "fixup":
                self.fixup = dfile["extras/fixup"][()]
            elif var == "imex_errors":
                self.imex_errors = dfile["extras/imex_errors"][()]
            elif var == "solver_failures":
                self.solver_failures = dfile["extras/solver_failures"][()]
            else:
                sys.exit("Invalid list of 'extras' provided. Exiting!")
