import numpy as np
import grid

diagnostic_dict = {
  # EMHD primitive quantities (unscaled)
  'q'  : lambda dump, **kwargs : calc_q(dump)
  'dP' : lambda dump, **kwargs : calc_dP(dump)
  
  # Some other basic quantities
  'pg'      : lambda dump, **kwargs : calc_pg(dump)
  'pb'      : lambda dump, **kwargs : calc_pb(dump)
  'ptot'    : lambda dump, **kwargs : calc_ptot(dump)
  'cs2'     : lambda dump, **kwargs : calc_cs2(dump)
  'beta'    : lambda dump, **kwargs : calc_plasma_beta(dump)
  'betaInv' : lambda dump, **kwargs : grid.inv_scalar(calc_plasma_beta(dump))
  'sigma'   : lambda dump, **kwargs : calc_sigma(dump)

  # Vectors, tensors and more
  'ucon'          : lambda dump, **kwargs : calc_ucon(dump),
  'ucov'          : lambda dump, G, **kwargs : grid.lower_vec(calc_ucon(dump), G)
  'bcon'          : lambda dump, **kwargs : calc_bcon(dump)
  'bcov'          : lambda dump, G, **kwargs : grid.lower_vec(calc_bcon(dump), G)
  'bsq'           : lambda dump, G, **kwargs : grid.dot_vec(calc_bcon(dump), grid.lower_vec(calc_bcon(dump), G))
  'lorentzFactor' : lambda dump, G, **kwargs : calc_lorentz_factor(dump, G)
  'Tcon'          : lambda dump, G, indices : calc_Tcon(dump, G, indices)
  'TconFl'        : lambda dump, G, indices : calc_TconFl(dump, G, indices)
  'TconEM'        : lambda dump, G, indices : calc_TconEM(dump, G, indices)
  'Tcov'          : lambda dump, G, indices : calc_Tcov(dump, G, indices)
  'TcovFl'        : lambda dump, G, indices : calc_TcovFl(dump, G, indices)
  'TcovEM'        : lambda dump, G, indices : calc_TcovEM(dump, G, indices)
  'Tmixed'        : lambda dump, indices, **kwargs : calc_Tcov(dump, indices)
  'TmixedFl'      : lambda dump, indices, **kwargs : calc_TmixedFl(dump, indices)
  'TmixedEM'      : lambda dump, indices, **kwargs : calc_TmixedEM(dump, indices)
  'MaxwellStress' : lambda dump, indices, **kwargs : calc_Maxwell_stress(dump, indices)
  'TPAKE'         : lambda dump, indices, **kwargs : calc_TPAKE(dump, indices)
  'TEN'           : lambda dump, indices, **kwargs : calc_T_thermal(dump, indices)

  # Fluxes
  'mdot'  : lambda dump, **kwargs : calc_mdot(dump)
  'phibh' : lambda dump, **kwargs : calc_phibh(dump)
  'ldot'  : lambda dump, **kwargs : calc_ldot(dump)
  'edot'  : lambda dump, **kwargs : calc_edot(dump)
}

def calc_q(dump):
  """Compute q from qtilde

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  """

  if dump.ho_conduction:
    cs2   = calc_cs2(dump)
    Theta = (dump.gam - 1.) * dump.u / dump.rho
    return dump.qtilde * np.sqrt(dump.rho * dump.conduction_alpha * cs2 * Theta**2)

  else:
    return dump.qtilde

def calc_dP(dump):
  """Compute dP from dPtilde

  :param dump: Fluid dump
  :type dump: :class:`harmless.fluid.FluidDump`
  """

  if dump.ho_viscosity:
    cs2   = calc_cs2(dump)
    Theta = (dump.gam - 1.) * dump.u / dump.rho
    return dump.dPtilde * np.sqrt(dump.rho * dump.conduction_alpha * cs2 * Theta**2)

  else:
    return dump.dPtilde