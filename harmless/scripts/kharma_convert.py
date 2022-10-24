import click, glob, os
from harmless import io
from harmless import parallelize as par

@click.command()
@click.option('-p', '--dumpsdir', default=None, help="Location of *.phdf file(s). Enter absolute path. If a single file is being converted, this will be the file name.")
@click.option('-o', '--outputdir', default=None, help="Location of converted *.h5 file(s). Enter absolute path.")
@click.option('-s', '--single_file', default=True, help="Are you converting just one file. Then the operation won't be parallelized even if asked to")
@click.option('-do_parallel', '--do_parallel', default=False, help="Parallelize the operation")
@click.option('-nthreads', '--nthreads', default=1, help="Number of threads/processes if the operation is parallelized")
@click.option('-pad', '--pad', default=0.4, help="Pad number of threads launched to avoid OOM")
def convert(dumspdir, outputdir, resume, single_file, do_parallel, nthreads):
  """Wrapper over convert_dump. Launches a single process or several based on the value of do_parallel. Runs on a single node nevertheless. This is fine because it's a one-off process.

  :param dumspdir: Location of native KHARMA fluid dumps (.phdf). Enter absolute path to correctly parse the path.
  :type dumspdir: str
  :param outputdir: Location where the iharm-format fluid dumps (.h5) will be stored. Enter absolute path to correctly parse the path.
  :type outputdir: str
  :param single_file: If set to true, the code expects only a single dump file to be converted. Doesn't parallelize the code.
  :type single_file: bool
  :param do_parallel: Should the operation be parallelized?
  :type do_parallel: bool
  :param nthreads: In the event the operation is being parallelized, how many child processes must be launched, or equivalently how many processes must be launched.
  If you get a OOM, reduce the padding `pad` which equivalently reduces the number of threads.
  :type nthreads: int
  :param pad: Padding provided to control number of threads launched.
  :type pad: float
  """

  # First check if dumpsdir exists. If not, fail loudly.
  if os.path.exists(dumpsdir):
    click.echo("Dump(s) location exists. Moving on to check output file/directory.")
  else:
    click.echo(f"{dumpsdir} doesn't exist. Check again. Aborting!")

  # If outputdir is not provided, let the user know that the converted file(s) will be stored in the same location as the original files
  if outputdir is None:
    click.echo(f"No output directory provided, will store converted files in the same location as {dumpsdir}")
  # Check if the location where .h5 files are to be stored exists. If not, make it.
  if os.path.exists(outputdir):
    click.echo("Output directory exists")
  else:
    click.echo("Output directory provided doesn't exist, making it.")
    os.makedirs(outputdir)

  if single_file:
    convert_dump(dumpsdir, outputdir)
  else:
    if do_parallel:
      # Control number of threads. May not necessarily work if padding is high.
      nthreads_max = par.calc_threads(pad)
      if nthreads > nthreads_max:
        nthreads = nthreads_max
    else:
      click.echo(f"do_parallel was set to {do_parallel}. Hence, launching just a single thread.")
      nthreads = 1

    # Make a list of files to be converted and the output files
    kharma_files = sorted(glob.glob(dumpsdir, 'torus.out0.*.phdf5'))
    iharm_files  = [f.replace(".phdf5", ".h5") for f in kharma_files]
    click.echo("Converting a total of {:d} files".format(len(kharma_files)))

    # launch process
    par.run_parallel(convert_dump, list(zip(kharma_files, iharm_files)), nthreads)
  

if __name__=="__main__":
  convert()
