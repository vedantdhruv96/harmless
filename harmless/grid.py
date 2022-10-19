import os
import numpy as np

class Grid:
  """ A class to generate simulation grid
  Stores native coordinates, spherical and Cartesian coordinates (when applicable), 
  metric components and metric determinant in native coordinates. Optionally, it can
  also cache the Christoffel symbol.

  ...
  Attributes
  
  """
  def sa