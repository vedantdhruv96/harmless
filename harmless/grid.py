import os, sys
import numpy as np

__all__ = ["Grid", "lower_vec", "raise_vec", "dot_vec", "inv_scalar"]


def lower_vec(vcon, G):
    """Lower a contravariant 4-vector using the metric.

    :param vcon: Contravariant 4-vector of shape (..., 4)
    :type vcon: numpy.ndarray

    :param G: Grid object providing gcov
    :type G: :class:`Grid`

    :return: Covariant 4-vector of shape (..., 4)
    :rtype: numpy.ndarray
    """
    return np.einsum("...ij,...j->...i", G.gcov, vcon)


def raise_vec(vcov, G):
    """Raise a covariant 4-vector using the inverse metric.

    :param vcov: Covariant 4-vector of shape (..., 4)
    :type vcov: numpy.ndarray

    :param G: Grid object providing gcon
    :type G: :class:`Grid`

    :return: Contravariant 4-vector of shape (..., 4)
    :rtype: numpy.ndarray
    """
    return np.einsum("...ij,...j->...i", G.gcon, vcov)


def dot_vec(vcov, vcon):
    """Contract a covariant and contravariant 4-vector.

    :param vcov: Covariant 4-vector of shape (..., 4)
    :type vcov: numpy.ndarray

    :param vcon: Contravariant 4-vector of shape (..., 4)
    :type vcon: numpy.ndarray

    :return: Scalar field
    :rtype: numpy.ndarray
    """
    return np.einsum("...i,...i->...", vcov, vcon)


def inv_scalar(x):
    """Return the element-wise reciprocal of a scalar field.

    :param x: Input array
    :type x: numpy.ndarray

    :return: 1/x
    :rtype: numpy.ndarray
    """
    return 1.0 / x


class Grid:
    """A class to generate simulation grid
    Stores native coordinates, spherical and Cartesian coordinates (when applicable),
    metric components and metric determinant in native coordinates.

    """

    def __init__(
        self, coord_sys, n1, n2, n3, a, r_out, x1min, x2min, x3min, x1max, x2max, x3max
    ):
        """Constructor method for Grid.
        Initializes the grid object

        :param coord_sys: The type of coordinate system for the Grid object,
        defaults to fmks
        :type coord_sys: string

        :param {n1,n2,n3}: Number of grid zones along X1, X2 and X3 respectively,
        defaults to {384,192,192}
        :type {n1,n2,n3}: int

        :param a: Black hole spin, defaults to 0.9375
        :type a: float

        :param r_out: Outer radius of simulation domain,
        defaults to 1000.0
        :type r_out: float

        :param {x1min,x2min,x3min}: Coordinates of first physical zone
        along X1, X2 and X3 respectively,
        defaults to {0.0,0.0,0.0}
        :type {x1min,x2min,x3min}: float

        :param {x1max,x2max,x3max}: Coordinates of last physical zone
        along X1, X2 and X3 respectively,
        defaults to {1.0,1.0,1.0}
        :type {x1max,x2max,x3max}: float
        """

        self.coord_sys = coord_sys
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        _valid = ("cartesian", "minkowski", "eks", "mks", "fmks")
        if self.coord_sys not in _valid:
            sys.exit(
                f"Not a valid coordinate system '{coord_sys}'. Valid options: {_valid}"
            )

        self.x1 = np.zeros((self.n1, self.n2, self.n3), dtype=float)
        self.x2 = np.zeros((self.n1, self.n2, self.n3), dtype=float)
        self.x3 = np.zeros((self.n1, self.n2, self.n3), dtype=float)

        self.r = np.zeros((self.n1, self.n2, self.n3), dtype=float)
        self.th = np.zeros((self.n1, self.n2, self.n3), dtype=float)
        self.phi = np.zeros((self.n1, self.n2, self.n3), dtype=float)

        self.gcov = np.zeros((self.n1, self.n2, self.n3, 4, 4), dtype=float)
        self.gcon = np.zeros((self.n1, self.n2, self.n3, 4, 4), dtype=float)
        self.gdet = np.zeros((self.n1, self.n2, self.n3), dtype=float)

        # Set grid widths
        if self.coord_sys == "cartesian" or self.coord_sys == "minkowski":
            self.startx1 = x1min
            self.startx2 = x2min
            self.startx3 = x3min

            self.dx1 = (x1max - self.startx1) / self.n1
            self.dx2 = (x2max - self.startx2) / self.n2
            self.dx3 = (x3max - self.startx3) / self.n3

        if (
            (self.coord_sys == "eks")
            or (self.coord_sys == "mks")
            or (self.coord_sys == "fmks")
        ):
            self.a = a
            self.r_out = r_out

            self.rEH = 1 + np.sqrt(1 - self.a * self.a)
            # Ensure 5 physical zones are present within the event horizon.
            # All HARM codes and its derivatives do this.
            r_in = np.exp(
                (self.n1 * np.log(self.rEH) / 5.5 - np.log(self.r_out))
                / (-1 + self.n1 / 5.5)
            )
            self.startx1 = np.log(r_in)
            self.startx2 = 0.0
            self.startx3 = 0.0

            self.dx1 = np.log(self.r_out / r_in) / self.n1
            self.dx2 = 1 / self.n2
            self.dx3 = 2 * np.pi / self.n3

        # Set native coordinates
        for i in range(self.n1):
            self.x1[i, :, :] = self.startx1 + ((i + 0.5) * self.dx1)
        for j in range(self.n2):
            self.x2[:, j, :] = self.startx2 + ((j + 0.5) * self.dx2)
        for k in range(self.n3):
            self.x3[:, :, k] = self.startx3 + ((k + 0.5) * self.dx3)

        # Initialize metric components and 'gdet'
        if self.coord_sys == "cartesian" or self.coord_sys == "minkowski":
            self.cartesian()
        elif self.coord_sys == "eks":
            self.eks()
        elif self.coord_sys == "mks":
            self.hslope = 0.3
            self.mks()
        elif self.coord_sys == "fmks":
            self.hslope = 0.3
            self.mks_smooth = 0.5
            self.poly_xt = 0.82
            self.poly_alpha = 14.0
            self.poly_norm = (
                0.5
                * np.pi
                / (1 + 1 / (self.poly_alpha + 1) * 1 / (self.poly_xt**self.poly_alpha))
            )
            self.Dx1 = self.x1[:, 0, 0] - self.startx1
            self.fmks()
        else:
            sys.exit("Not a valid coordinate system. Exiting!")

        self.lapse = 1.0 / np.sqrt(-self.gcon[Ellipsis, 0, 0])

    def gcov_ks(self):
        """Generates the KS metric.
        Need it to compute the metric components in native coordinates.
        """

        gcov_ks = np.zeros((self.n1, self.n2, self.n3, 4, 4), dtype=float)
        sigma = self.r**2 + ((self.a**2) * np.cos(self.th) ** 2)

        gcov_ks[Ellipsis, 0, 0] = -1 + (2 * self.r / sigma)
        gcov_ks[Ellipsis, 0, 1] = gcov_ks[Ellipsis, 1, 0] = 2 * self.r / sigma
        gcov_ks[Ellipsis, 0, 3] = gcov_ks[Ellipsis, 3, 0] = (
            -2 * self.a * self.r * np.sin(self.th) ** 2 / sigma
        )
        gcov_ks[Ellipsis, 1, 1] = 1 + (2 * self.r / sigma)
        gcov_ks[Ellipsis, 1, 3] = gcov_ks[Ellipsis, 3, 1] = (
            -self.a * np.sin(self.th) ** 2 * (1 + (2 * self.r / sigma))
        )
        gcov_ks[Ellipsis, 2, 2] = sigma
        gcov_ks[Ellipsis, 3, 3] = np.sin(self.th) ** 2 * (
            sigma + self.a**2 * np.sin(self.th) ** 2 * (1 + (2 * self.r / sigma))
        )

        return gcov_ks

    def dxdX(self):
        """Generates the transformation matrix to transform covariant
        metric from KS to native coordinates.
        """
        dxdX = np.zeros((self.n1, self.n2, self.n3, 4, 4), dtype=float)

        if self.coord_sys == "eks":
            dxdX[Ellipsis, 0, 0] = dxdX[Ellipsis, 3, 3] = 1.0
            dxdX[Ellipsis, 1, 1] = np.exp(self.x1)
            dxdX[Ellipsis, 2, 2] = np.pi

        elif self.coord_sys == "mks":
            dxdX[Ellipsis, 0, 0] = dxdX[Ellipsis, 3, 3] = 1.0
            dxdX[Ellipsis, 1, 1] = np.exp(self.x1)
            dxdX[Ellipsis, 2, 2] = np.pi + (1 - self.hslope) * np.pi * np.cos(
                2 * np.pi * self.x2
            )

        elif self.coord_sys == "fmks":
            D = (np.pi * self.poly_xt**self.poly_alpha) / (
                2 * self.poly_xt**self.poly_alpha + (2 / (1 + self.poly_alpha))
            )
            theta_g = (np.pi * self.x2) + ((1 - self.hslope) / 2) * (
                np.sin(2 * np.pi * self.x2)
            )
            theta_j = (
                D
                * (2 * self.x2 - 1)
                * (
                    1
                    + (((2 * self.x2 - 1) / self.poly_xt) ** self.poly_alpha)
                    / (1 + self.poly_alpha)
                )
                + np.pi / 2
            )
            derv_theta_g = np.pi + (1 - self.hslope) * np.pi * np.cos(
                2 * np.pi * self.x2
            )
            derv_theta_j = (
                2
                * self.poly_alpha
                * D
                * (2 * self.x2 - 1)
                * ((2 * self.x2 - 1) / self.poly_xt) ** (self.poly_alpha - 1)
            ) / (self.poly_xt * (self.poly_alpha + 1)) + 2 * D * (
                1
                + (((2 * self.x2 - 1) / self.poly_xt) ** self.poly_alpha)
                / (self.poly_alpha + 1)
            )

            dxdX[Ellipsis, 0, 0] = dxdX[Ellipsis, 3, 3] = 1.0
            dxdX[Ellipsis, 1, 1] = np.exp(self.x1)
            dxdX[Ellipsis, 2, 1] = (
                -self.mks_smooth
                * np.exp(-self.mks_smooth * self.Dx1[:, np.newaxis, np.newaxis])
                * (theta_j - theta_g)
            )
            dxdX[Ellipsis, 2, 2] = derv_theta_g + np.exp(
                -self.mks_smooth * self.Dx1[:, np.newaxis, np.newaxis]
            ) * (derv_theta_j - derv_theta_g)

        return dxdX

    def cartesian(self):
        """Cartesian coordinates method.
        Calculate metric components and metric determinant.
        """
        self.gcov[Ellipsis, 0, 0] = self.gcon[Ellipsis, 0, 0] = -1
        self.gcov[Ellipsis, 1, 1] = self.gcon[Ellipsis, 1, 1] = 1
        self.gcov[Ellipsis, 2, 2] = self.gcon[Ellipsis, 2, 2] = 1
        self.gcov[Ellipsis, 3, 3] = self.gcon[Ellipsis, 3, 3] = 1
        self.gdet = 1.0

    def eks(self):
        """EKS coordinates method.
        Calculate metric components and metric determinant.
        """
        self.r = np.exp(self.x1)
        self.th = np.pi * self.x2
        self.phi = self.x3

        gcov_ks = self.gcov_ks()
        dxdX = self.dxdX()
        self.gcov = np.einsum(
            "ijkbn,ijkmb->ijkmn", dxdX, np.einsum("ijkam,ijkab->ijkmb", dxdX, gcov_ks)
        )
        self.gcon = np.linalg.inv(self.gcov)
        self.gdet = np.sqrt(-np.linalg.det(self.gcov))

    def mks(self):
        """MKS coordinates method.
        Calculate metric components and metric determinant.
        """
        self.r = np.exp(self.x1)
        self.th = (np.pi * self.x2) + ((1 - self.hslope) / 2) * np.sin(
            2 * np.pi * self.x2
        )
        self.phi = self.x3

        gcov_ks = self.gcov_ks()
        dxdX = self.dxdX()
        self.gcov = np.einsum(
            "ijkbn,ijkmb->ijkmn", dxdX, np.einsum("ijkam,ijkab->ijkmb", dxdX, gcov_ks)
        )
        self.gcon = np.linalg.inv(self.gcov)
        self.gdet = np.sqrt(-np.linalg.det(self.gcov))

    def fmks(self):
        """FMKS coordinates method.
        Calculate metric components and metric determinant
        """
        D = (np.pi * self.poly_xt**self.poly_alpha) / (
            2 * self.poly_xt**self.poly_alpha + (2 / (1 + self.poly_alpha))
        )
        theta_g = (np.pi * self.x2) + ((1 - self.hslope) / 2) * np.sin(
            2 * np.pi * self.x2
        )
        theta_j = (
            D
            * (2 * self.x2 - 1)
            * (
                1
                + (((2 * self.x2 - 1) / self.poly_xt) ** self.poly_alpha)
                / (1 + self.poly_alpha)
            )
            + np.pi / 2
        )
        self.r = np.exp(self.x1)
        self.th = theta_g + np.exp(
            -self.mks_smooth * self.Dx1[:, np.newaxis, np.newaxis]
        ) * (theta_j - theta_g)
        self.phi = self.x3

        gcov_ks = self.gcov_ks()
        dxdX = self.dxdX()
        self.gcov = np.einsum(
            "ijkbn,ijkmb->ijkmn", dxdX, np.einsum("ijkam,ijkab->ijkmb", dxdX, gcov_ks)
        )
        self.gcon = np.linalg.inv(self.gcov)
        self.gdet = np.sqrt(-np.linalg.det(self.gcov))
