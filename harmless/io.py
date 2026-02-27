import h5py
import numpy as np

__all__ = ["write_dump", "write_grid"]


def write_dump(dump, fname):
  """Write a FluidDump object to an iharm-format HDF5 file.

  :param dump: The fluid dump object
  :type dump: :class:`harmless.fluid.FluidDump`
  :param fname: Output filename
  :type fname: str
  :raises NotImplementedError: This function is not yet implemented.
  """
  raise NotImplementedError("write_dump is not yet implemented.")


def write_grid(G, fname):
  """Save the grid object

  :param G: The grid object
  :type G: :class:`harmless.grid.Grid`
  :param fname: Grid filename
  :type fname: str

  """
  gfile = h5py.File(fname, 'w')
  if (G.coord_sys == 'cartesian') or (G.coord_sys == 'minkowski'):
    gfile['X'] = G.x1
    gfile['Y'] = G.x2
    gfile['Z'] = G.x3
  else:
    gfile['X'] = G.r * np.sin(G.th) * np.cos(G.phi)
    gfile['Y'] = G.r * np.sin(G.th) * np.sin(G.phi)
    gfile['Z'] = G.r * np.cos(G.th)

  gfile['r']   = G.r
  gfile['th']  = G.th
  gfile['phi'] = G.phi

  gfile['X1'] = G.x1
  gfile['X2'] = G.x2
  gfile['X3'] = G.x3

  gfile['gcov']  = G.gcov
  gfile['gcon']  = G.gcon
  gfile['gdet']  = G.gdet
  gfile['lapse'] = G.lapse
  gfile.close()
