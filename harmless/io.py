import h5py
import numpy as np


def load_dump():
  pass

def write_grid(G, fname):
  """Save the grid object
  """
  gfile = h5py.File(fname, 'w')
  if (G.coord_sys == 'cartesian') or (G.coord_sys == 'minkowski'):
    gfile['X'] = G.x1
    gfile['Y'] = G.x2
    gfile['Z'] = G.x3
  else:
    gfile['X'] = G.r * np.sin(G.th) * np.cos(G.phi)
    gfile['Y'] = G.r * np.sin(G.th) * np.sin(G.phi)
    gfile['Z'] = G.r * np.sin(G.phi)
  
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