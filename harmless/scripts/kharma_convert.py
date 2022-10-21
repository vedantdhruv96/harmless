import click
from harmless import io
from harmless import parallelize as par

@click.command()
@click.option('-p', '--dumpsdir', default=None, help="Location of *.phdf file(s). Enter absolute path.")
@click.option('-o', '--outputdir', default=None, help="Location of converted *.h5 file(s). Enter absolute path.")
@click.option('-r', '--resume', default=True, help="Only convert those .phdf files that are not present in outputdir")
@click.option('-s', '--single_file', default=True, help="Are you converting just one file. Then the operation won't be parallelized even if asked to")
@click.option('-do_parallel', '--do_parallel', default=False, help="Parallelize the operation")
@click.option('-nthreads', '--nthreads', default=1, help="Number of threads/processes if the operation is parallelized")
def convert(dumspdir, outputdir, resume, single_file, do_parallel, nthreads):
  """Wrapper over convert_dump. Launches a single process or several based on the value of do_parallel

  :param dumspdir: Location of native KHARMA fluid dumps (.phdf). Enter absolute path to correctly parse the path.
  :type dumspdir: str
  :param outputdir: Location where the iharm-format fluid dumps (.h5) will be stored. Enter absolute path to correctly parse the path.
  :type outputdir: str
  :param resume: If set to True, will convert only those .phdf files that have not been convert to .h5 already.
  :type resume: bool
  :param single_file: If set to true, the code expects only a single dump file to be converted. Doesn't parallelize the code.
  :type single_file: bool
  :param do_parallel: Should the operation be parallelized?
  :type do_parallel: bool
  :param nthreads: In the event the operation is being parallelized, how many child processes must be launched, or equivalently how many processes must be launched.
  :type nthreads: int
  """
  pass

if __name__=="__main__":
  convert()
