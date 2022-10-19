#!/usr/bin/env python3

import os, click
from harmless import grid

@click.command()
@click.argument('coord_sys', nargs=1, default='fmks')
@click.argument('n1', nargs=1, default=384, type=int)
@click.argument('n2', nargs=1, default=192, type=int)
@click.argument('n3', nargs=1, default=192, type=int)
@click.argument('a', nargs=1, default=0.9375, type=float)
@click.argument('r_out', nargs=1, default=1000, type=float)
@click.option('--out', 'dir', default=os.environ.get("PWD"), show_default=True, help="Output file location, enter absolute path")
def make_grid(coord_sys, n1, n2, n3, a, r_out, dir):
  """This function creates a grid file (grid.h5) and stores it in a user provided location. 
  If no location is provided, it defaults to saving the grid file in the location the script
  is executed from.

  Arguments:

  \b
    coord_sys (STRING): The simulation coordinate system {minkoski, cartesian, eks, mks, fmks; DEFAULT: fmks}
  \b
    n1 (INT)          : Number of physical zones along X1 {DEFAULT: 384}
  \b
    n2 (INT)          : Number of physical zones along X2 {DEFAULT: 192}
  \b
    n3 (INT)          : Number of physical zones along X3 {DEFAULT: 192}
  \b
    a (FLOAT)         : Black hole spin {DEFAULT: 0.9375}
  \b
    r_out (FLOAT)     : Outer radius of simulation domain {DEFAULT: 1000}
  """
  click.echo(f"Generating {coord_sys} grid of size {n1}x{n2}x{n3} with r_out={r_out}")
  grid_file = grid.Grid(coord_sys, n1, n2, n3, a, r_out)


if __name__ == "__main__":
  make_grid()