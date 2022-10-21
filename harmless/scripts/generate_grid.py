#!/usr/bin/env python3

import os, click
from harmless import grid, io

@click.command()
@click.argument('coord_sys', nargs=1, default='fmks')
@click.argument('n1', nargs=1, default=384, type=int)
@click.argument('n2', nargs=1, default=192, type=int)
@click.argument('n3', nargs=1, default=192, type=int)
@click.argument('a', nargs=1, default=0.9375, type=float)
@click.argument('r_out', nargs=1, default=1000, type=float)
@click.option('-x1min', '--x1min', default=0.0, show_default=True, help="X1 coordinate of first physical zone")
@click.option('-x2min', '--x2min', default=0.0, show_default=True, help="X2 coordinate of first physical zone")
@click.option('-x3min', '--x3min', default=0.0, show_default=True, help="X3 coordinate of first physical zone")
@click.option('-x1max', '--x1max', default=1.0, show_default=True, help="X1 coordinate of last physical zone")
@click.option('-x2max', '--x2max', default=1.0, show_default=True, help="X2 coordinate of last physical zone")
@click.option('-x3max', '--x3max', default=1.0, show_default=True, help="X3 coordinate of last physical zone")
@click.option('-out', '--fname', default=os.path.join(os.environ.get("PWD"), 'grid.h5'), show_default=True, help="Output file name, enter absolute path")
def make_grid(coord_sys, n1, n2, n3, a, r_out, x1min, x2min, x3min, x1max, x2max, x3max, fname):
  """This function creates a grid file (grid.h5) and stores it in a user provided location. 
  If no location is provided, it defaults to saving the grid file in the location the script
  is executed from. x1min, x1max, ... will be considered only if the coordinate system is 'cartesian' or 'minkowski'.

  :param coord_sys: The simulation coordinate system {minkoski, cartesian, eks, mks, fmks}, defaults to fmks
  :type coord_sys: string
  :param n1: Number of physical zones along X1, defaults to 384
  :type n1: int
  :param n2: Number of physical zones along X2, defaults to 192
  :type n2: int
  :param n3: Number of physical zones along X3, defaults to 192
  :type n3: int
  :param a: Black hole spin, defaults to 0.9375
  :type a: float
  :param r_out: Outer radius of simulation domain, defaults to 1000.
  :type r_out: float
  :param x1min: X1 coordinate of first physical zone, defaults to 0.0
  :type x1min: float
  :param x2min: X2 coordinate of first physical zone, defaults to 0.0
  :type x2min: float
  :param x3min: X3 coordinate of first physical zone, defaults to 0.0
  :type x3min: float
  :param x1max: X1 coordinate of last physical zone, defaults to 1.0
  :type x1max: float
  :param x2max: X2 coordinate of last physical zone, defaults to 1.0
  :type x2max: float
  :param x3max: X3 coordinate of last physical zone, defaults to 1.0
  :type x3max: float
  :param fname: Output file name, enter absolute path, defaults to $PWD/grid.h5
  :type fname: str
  """
  click.echo(f"Generating {coord_sys} grid of size {n1}x{n2}x{n3} for spin {a} with r_out={r_out}")
  grid_file = grid.Grid(coord_sys, n1, n2, n3, a, r_out, x1min, x2min, x3min, x1max, x2max, x3max)
  io.write_grid(grid_file, fname)


if __name__ == "__main__":
  make_grid()