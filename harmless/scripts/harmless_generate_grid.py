#!/usr/bin/env python3

import click
from harmless import grid

@click.command()
@click.argument('coord_sys', nargs=1, default='fmks')
@click.argument('n1', nargs=1, default=384, type=int)
@click.argument('n2', nargs=1, default=192, type=int)
@click.argument('n3', nargs=1, default=192, type=int)
@click.argument('a', nargs=1, default=0.9375, type=float)
@click.argument('r_out', nargs=1, default=1000, type=float)
def make_grid(coord_sys, n1, n2, n3, a, r_out):
  pass

if __name__ == "__main__":
  make_grid()