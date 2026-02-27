import numpy as np
from harmless import grid

__all__ = ["diagnostic_dict"]

# Map of string keys to lambda functions for computing derived quantities.
# All lambdas have signature: (dump, G=None, indices=None, **kwargs)
# G is a :class:`harmless.grid.Grid` object (required for metric-dependent quantities).
# indices is a tuple (mu, nu) selecting a tensor component; None returns the full array.
diagnostic_dict = {
  # EMHD primitive quantities (unscaled)
  'q'  : lambda dump, G=None, indices=None, **kwargs: calc_q(dump),
  'dP' : lambda dump, G=None, indices=None, **kwargs: calc_dP(dump),

  # Thermodynamic scalars
  'pg'      : lambda dump, G=None, indices=None, **kwargs: calc_pg(dump),
  'pb'      : lambda dump, G=None, indices=None, **kwargs: calc_pb(dump, G),
  'ptot'    : lambda dump, G=None, indices=None, **kwargs: calc_ptot(dump, G),
  'cs2'     : lambda dump, G=None, indices=None, **kwargs: calc_cs2(dump),
  'beta'    : lambda dump, G=None, indices=None, **kwargs: calc_plasma_beta(dump, G),
  'betaInv' : lambda dump, G=None, indices=None, **kwargs: grid.inv_scalar(calc_plasma_beta(dump, G)),
  'sigma'   : lambda dump, G=None, indices=None, **kwargs: calc_sigma(dump, G),

  # 4-vectors
  'ucon'          : lambda dump, G=None, indices=None, **kwargs: calc_ucon(dump, G),
  'ucov'          : lambda dump, G=None, indices=None, **kwargs: grid.lower_vec(calc_ucon(dump, G), G),
  'bcon'          : lambda dump, G=None, indices=None, **kwargs: calc_bcon(dump, G),
  'bcov'          : lambda dump, G=None, indices=None, **kwargs: grid.lower_vec(calc_bcon(dump, G), G),
  'bsq'           : lambda dump, G=None, indices=None, **kwargs: calc_bsq(dump, G),
  'lorentzFactor' : lambda dump, G=None, indices=None, **kwargs: calc_lorentz_factor(dump, G),

  # Stress-energy tensor (contravariant T^{mu nu})
  'Tcon'   : lambda dump, G=None, indices=None, **kwargs: calc_Tcon(dump, G, indices),
  'TconFl' : lambda dump, G=None, indices=None, **kwargs: calc_TconFl(dump, G, indices),
  'TconEM' : lambda dump, G=None, indices=None, **kwargs: calc_TconEM(dump, G, indices),

  # Stress-energy tensor (covariant T_{mu nu})
  'Tcov'   : lambda dump, G=None, indices=None, **kwargs: calc_Tcov(dump, G, indices),
  'TcovFl' : lambda dump, G=None, indices=None, **kwargs: calc_TcovFl(dump, G, indices),
  'TcovEM' : lambda dump, G=None, indices=None, **kwargs: calc_TcovEM(dump, G, indices),

  # Mixed stress-energy tensor (T^mu_nu)
  'Tmixed'      : lambda dump, G=None, indices=None, **kwargs: calc_Tmixed(dump, G, indices),
  'TmixedFl'    : lambda dump, G=None, indices=None, **kwargs: calc_TmixedFl(dump, G, indices),
  'TmixedEM'    : lambda dump, G=None, indices=None, **kwargs: calc_TmixedEM(dump, G, indices),
  'MaxwellStress': lambda dump, G=None, indices=None, **kwargs: calc_Maxwell_stress(dump, G, indices),
  'TPAKE'       : lambda dump, G=None, indices=None, **kwargs: calc_TPAKE(dump, G, indices),
  'TEN'         : lambda dump, G=None, indices=None, **kwargs: calc_T_thermal(dump, G, indices),

  # Radial flux profiles (shape: n1)
  'mdot'  : lambda dump, G=None, indices=None, **kwargs: calc_mdot(dump, G),
  'phibh' : lambda dump, G=None, indices=None, **kwargs: calc_phibh(dump, G),
  'ldot'  : lambda dump, G=None, indices=None, **kwargs: calc_ldot(dump, G),
  'edot'  : lambda dump, G=None, indices=None, **kwargs: calc_edot(dump, G),
}


# ---------------------------------------------------------------------------
# EMHD quantities
# ---------------------------------------------------------------------------

def calc_q(dump):
  """Compute heat flux q from qtilde. Applies EMHD higher-order correction when enabled.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :return: Heat flux q
  :rtype: numpy.ndarray
  """
  if dump.ho_conduction:
    cs2   = calc_cs2(dump)
    Theta = (dump.gam - 1.) * dump.u / dump.rho
    return dump.qtilde * np.sqrt(dump.rho * dump.conduction_alpha * cs2 * Theta**2)
  else:
    return dump.qtilde


def calc_dP(dump):
  """Compute viscous pressure anisotropy dP from dPtilde. Applies EMHD higher-order correction when enabled.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :return: Pressure anisotropy dP
  :rtype: numpy.ndarray
  """
  if dump.ho_viscosity:
    cs2   = calc_cs2(dump)
    Theta = (dump.gam - 1.) * dump.u / dump.rho
    return dump.dPtilde * np.sqrt(dump.rho * dump.viscosity_alpha * cs2 * Theta**2)
  else:
    return dump.dPtilde


# ---------------------------------------------------------------------------
# Thermodynamic scalars
# ---------------------------------------------------------------------------

def calc_pg(dump):
  """Compute gas pressure: p_g = (gam - 1) * u.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :return: Gas pressure
  :rtype: numpy.ndarray
  """
  return (dump.gam - 1.) * dump.u


def calc_bsq(dump, G):
  """Compute magnetic pressure proxy b^2 = b_mu b^mu.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: b^2
  :rtype: numpy.ndarray
  """
  bcon = calc_bcon(dump, G)
  bcov = grid.lower_vec(bcon, G)
  return grid.dot_vec(bcov, bcon)


def calc_pb(dump, G):
  """Compute magnetic pressure: p_b = b^2 / 2.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Magnetic pressure
  :rtype: numpy.ndarray
  """
  return calc_bsq(dump, G) / 2.


def calc_ptot(dump, G):
  """Compute total pressure: p_tot = p_g + p_b.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Total pressure
  :rtype: numpy.ndarray
  """
  return calc_pg(dump) + calc_pb(dump, G)


def calc_cs2(dump):
  """Compute adiabatic sound speed squared: c_s^2 = gam * p_g / (rho + gam * u).

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :return: Sound speed squared
  :rtype: numpy.ndarray
  """
  pg = calc_pg(dump)
  return dump.gam * pg / (dump.rho + dump.gam * dump.u)


def calc_plasma_beta(dump, G):
  """Compute plasma beta: beta = p_g / p_b.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Plasma beta
  :rtype: numpy.ndarray
  """
  return calc_pg(dump) / calc_pb(dump, G)


def calc_sigma(dump, G):
  """Compute magnetization: sigma = b^2 / rho.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Magnetization sigma
  :rtype: numpy.ndarray
  """
  return calc_bsq(dump, G) / dump.rho


# ---------------------------------------------------------------------------
# 4-vectors
# ---------------------------------------------------------------------------

def calc_ucon(dump, G):
  """Compute the contravariant 4-velocity u^mu from HARM primitive velocities.

  HARM stores primitive velocities as ``u_tilde^i = u^i + beta^i * u^0``.
  The Lorentz factor is ``gamma = sqrt(1 + g_{ij} u_tilde^i u_tilde^j)``
  and then ``u^0 = gamma / alpha``, ``u^i = u_tilde^i - beta^i * u^0``.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object providing the metric.
  :type G: :class:`harmless.grid.Grid`
  :return: 4-velocity shape (n1, n2, n3, 4)
  :rtype: numpy.ndarray
  """
  # Compute q^2 = g_{ij} u_tilde^i u_tilde^j  (3-metric spatial sum)
  qsq = (G.gcov[..., 1, 1] * dump.u1**2
       + G.gcov[..., 2, 2] * dump.u2**2
       + G.gcov[..., 3, 3] * dump.u3**2
       + 2. * (G.gcov[..., 1, 2] * dump.u1 * dump.u2
             + G.gcov[..., 1, 3] * dump.u1 * dump.u3
             + G.gcov[..., 2, 3] * dump.u2 * dump.u3))

  gamma = np.sqrt(1. + qsq)

  ucon      = np.zeros((dump.n1, dump.n2, dump.n3, 4))
  ucon[..., 0] = gamma / G.lapse

  # Shift vector beta^i = -gcon^{0i} / gcon^{00}
  beta1 = -G.gcon[..., 0, 1] / G.gcon[..., 0, 0]
  beta2 = -G.gcon[..., 0, 2] / G.gcon[..., 0, 0]
  beta3 = -G.gcon[..., 0, 3] / G.gcon[..., 0, 0]

  ucon[..., 1] = dump.u1 - beta1 * ucon[..., 0]
  ucon[..., 2] = dump.u2 - beta2 * ucon[..., 0]
  ucon[..., 3] = dump.u3 - beta3 * ucon[..., 0]

  return ucon


def calc_lorentz_factor(dump, G):
  """Compute the Lorentz factor gamma = alpha * u^0.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Lorentz factor
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  return G.lapse * ucon[..., 0]


def calc_bcon(dump, G):
  """Compute the contravariant magnetic 4-vector b^mu.

  Uses the HARM convention:
  ``b^0 = u_i B^i / u^0`` and ``b^i = (B^i + b^0 u^i) / u^0``
  where ``u_i`` are the covariant spatial components of the 4-velocity.

  :param dump: Fluid dump storing primitive magnetic fields B1, B2, B3
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Magnetic 4-vector shape (n1, n2, n3, 4)
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  ucov = grid.lower_vec(ucon, G)

  # b^0 = u_i B^i  (spatial components only, indexed 1..3)
  b0 = ucov[..., 1] * dump.B1 + ucov[..., 2] * dump.B2 + ucov[..., 3] * dump.B3

  bcon      = np.zeros((dump.n1, dump.n2, dump.n3, 4))
  bcon[..., 0] = b0
  bcon[..., 1] = (dump.B1 + b0 * ucon[..., 1]) / ucon[..., 0]
  bcon[..., 2] = (dump.B2 + b0 * ucon[..., 2]) / ucon[..., 0]
  bcon[..., 3] = (dump.B3 + b0 * ucon[..., 3]) / ucon[..., 0]

  return bcon


# ---------------------------------------------------------------------------
# Stress-energy tensor helpers
# ---------------------------------------------------------------------------

def _select(arr, indices):
  """Return ``arr[..., mu, nu]`` if indices is a 2-tuple, else the full array."""
  if indices is not None:
    mu, nu = indices
    return arr[..., mu, nu]
  return arr


def _build_Tcon(dump, G):
  """Build the full contravariant MHD stress-energy tensor T^{mu nu}.

  ``T^{mu nu} = (rho + u + pg + bsq) u^mu u^nu + (pg + bsq/2) g^{mu nu} - b^mu b^nu``

  :return: Shape (n1, n2, n3, 4, 4)
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  bcon = calc_bcon(dump, G)
  bsq  = calc_bsq(dump, G)
  pg   = calc_pg(dump)

  w    = dump.rho + dump.u + pg + bsq        # enthalpy + b^2
  ptot = pg + bsq / 2.

  # outer product: shape (n1, n2, n3, 4, 4)
  T = (w[..., np.newaxis, np.newaxis] * ucon[..., :, np.newaxis] * ucon[..., np.newaxis, :]
     + ptot[..., np.newaxis, np.newaxis] * G.gcon
     - bcon[..., :, np.newaxis] * bcon[..., np.newaxis, :])
  return T


def _build_TconFl(dump, G):
  """Build the fluid part of the contravariant stress-energy tensor.

  ``T_Fl^{mu nu} = (rho + u + pg) u^mu u^nu + pg * g^{mu nu}``

  :return: Shape (n1, n2, n3, 4, 4)
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  pg   = calc_pg(dump)
  wfl  = dump.rho + dump.u + pg

  T = (wfl[..., np.newaxis, np.newaxis] * ucon[..., :, np.newaxis] * ucon[..., np.newaxis, :]
     + pg[..., np.newaxis, np.newaxis] * G.gcon)
  return T


def _build_TconEM(dump, G):
  """Build the electromagnetic part of the contravariant stress-energy tensor.

  ``T_EM^{mu nu} = bsq * u^mu u^nu + (bsq/2) * g^{mu nu} - b^mu b^nu``

  :return: Shape (n1, n2, n3, 4, 4)
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  bcon = calc_bcon(dump, G)
  bsq  = calc_bsq(dump, G)

  T = (bsq[..., np.newaxis, np.newaxis] * ucon[..., :, np.newaxis] * ucon[..., np.newaxis, :]
     + (bsq / 2.)[..., np.newaxis, np.newaxis] * G.gcon
     - bcon[..., :, np.newaxis] * bcon[..., np.newaxis, :])
  return T


def calc_Tcon(dump, G, indices=None):
  """Contravariant MHD stress-energy tensor T^{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T^{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  return _select(_build_Tcon(dump, G), indices)


def calc_TconFl(dump, G, indices=None):
  """Fluid part of the contravariant stress-energy tensor T_Fl^{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_Fl^{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  return _select(_build_TconFl(dump, G), indices)


def calc_TconEM(dump, G, indices=None):
  """Electromagnetic part of the contravariant stress-energy tensor T_EM^{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_EM^{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  return _select(_build_TconEM(dump, G), indices)


def calc_Tcov(dump, G, indices=None):
  """Covariant MHD stress-energy tensor T_{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  Tcon = _build_Tcon(dump, G)
  # T_{mu nu} = g_{mu alpha} g_{nu beta} T^{alpha beta}
  Tcov = np.einsum('...im,...jn,...mn->...ij', G.gcov, G.gcov, Tcon)
  return _select(Tcov, indices)


def calc_TcovFl(dump, G, indices=None):
  """Covariant fluid part of the stress-energy tensor T_Fl_{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_Fl_{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  TconFl = _build_TconFl(dump, G)
  TcovFl = np.einsum('...im,...jn,...mn->...ij', G.gcov, G.gcov, TconFl)
  return _select(TcovFl, indices)


def calc_TcovEM(dump, G, indices=None):
  """Covariant EM part of the stress-energy tensor T_EM_{mu nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_EM_{mu nu} or a specific component
  :rtype: numpy.ndarray
  """
  TconEM = _build_TconEM(dump, G)
  TcovEM = np.einsum('...im,...jn,...mn->...ij', G.gcov, G.gcov, TconEM)
  return _select(TcovEM, indices)


def calc_Tmixed(dump, G, indices=None):
  """Mixed MHD stress-energy tensor T^mu_nu = T^{mu alpha} g_{alpha nu}.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  Tcon    = _build_Tcon(dump, G)
  Tmixed  = np.einsum('...mn,...nj->...mj', Tcon, G.gcov)
  return _select(Tmixed, indices)


def calc_TmixedFl(dump, G, indices=None):
  """Mixed fluid part of the stress-energy tensor T_Fl^mu_nu.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_Fl^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  TconFl    = _build_TconFl(dump, G)
  TmixedFl  = np.einsum('...mn,...nj->...mj', TconFl, G.gcov)
  return _select(TmixedFl, indices)


def calc_TmixedEM(dump, G, indices=None):
  """Mixed EM part of the stress-energy tensor T_EM^mu_nu.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_EM^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  TconEM    = _build_TconEM(dump, G)
  TmixedEM  = np.einsum('...mn,...nj->...mj', TconEM, G.gcov)
  return _select(TmixedEM, indices)


def calc_Maxwell_stress(dump, G, indices=None):
  """Maxwell (EM) contribution to the mixed stress tensor.
  Alias for :func:`calc_TmixedEM`.

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_EM^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  return calc_TmixedEM(dump, G, indices)


def calc_TPAKE(dump, G, indices=None):
  """Poynting + kinetic (fluid) contribution to the mixed stress tensor.
  Returns T_total^mu_nu minus the thermal pressure part T_thermal^mu_nu.
  This is the Poynting-And-Kinetic-Energy combination.

  When ``indices=(1,3)`` this gives the r-phi component (angular momentum flux density).

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: (T_total - T_thermal)^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  Tmixed         = calc_Tmixed(dump, G)
  Tmixed_thermal = calc_T_thermal(dump, G)
  return _select(Tmixed - Tmixed_thermal, indices)


def calc_T_thermal(dump, G, indices=None):
  """Thermal gas-pressure contribution to the mixed stress-energy tensor.

  ``T_thermal^mu_nu = pg * delta^mu_nu`` (thermal pressure times identity).

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :param indices: Component tuple (mu, nu) or None for full tensor
  :type indices: tuple, optional
  :return: T_thermal^mu_nu or a specific component
  :rtype: numpy.ndarray
  """
  pg = calc_pg(dump)
  # T_thermal^mu_nu = pg * g^{mu alpha} g_{alpha nu} = pg * delta^mu_nu
  # Build identity-like shape (n1, n2, n3, 4, 4)
  Tthermal = pg[..., np.newaxis, np.newaxis] * np.eye(4)
  return _select(Tthermal, indices)


# ---------------------------------------------------------------------------
# Radial flux profiles
# ---------------------------------------------------------------------------

def calc_mdot(dump, G):
  """Compute the mass accretion rate as a function of radius.

  ``Mdot(r) = -sum_{theta,phi} rho * u^r * sqrt(-g) * dX2 * dX3``

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Mdot profile of shape (n1,)
  :rtype: numpy.ndarray
  """
  ucon = calc_ucon(dump, G)
  integrand = dump.rho * ucon[..., 1] * G.gdet
  return -np.sum(integrand, axis=(1, 2)) * G.dx2 * G.dx3


def calc_phibh(dump, G):
  """Compute the magnetic flux through a sphere as a function of radius.

  ``Phi(r) = 0.5 * sum_{theta,phi} |B^r| * sqrt(-g) * dX2 * dX3``

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Phi profile of shape (n1,)
  :rtype: numpy.ndarray
  """
  integrand = np.abs(dump.B1) * G.gdet
  return 0.5 * np.sum(integrand, axis=(1, 2)) * G.dx2 * G.dx3


def calc_ldot(dump, G):
  """Compute the angular momentum flux as a function of radius.

  ``Ldot(r) = -sum_{theta,phi} T^r_phi * sqrt(-g) * dX2 * dX3``

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Ldot profile of shape (n1,)
  :rtype: numpy.ndarray
  """
  Tmixed    = calc_Tmixed(dump, G)
  integrand = Tmixed[..., 1, 3] * G.gdet
  return -np.sum(integrand, axis=(1, 2)) * G.dx2 * G.dx3


def calc_edot(dump, G):
  """Compute the energy flux as a function of radius.

  ``Edot(r) = -sum_{theta,phi} T^r_t * sqrt(-g) * dX2 * dX3``

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  :param G: Grid object
  :type G: :class:`harmless.grid.Grid`
  :return: Edot profile of shape (n1,)
  :rtype: numpy.ndarray
  """
  Tmixed    = calc_Tmixed(dump, G)
  integrand = Tmixed[..., 1, 0] * G.gdet
  return -np.sum(integrand, axis=(1, 2)) * G.dx2 * G.dx3
