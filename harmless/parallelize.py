import multiprocessing as mp
import psutil

__all__ = ["calc_threads", "run_parallel"]


def calc_threads(pad=0.4):
    """Compute the total number of threads / processes to launch based on,
    (i)  the total number of physical cores on a node
    (ii) the padding provided

    :param pad: padding, used as <cpu cores> * <pad>, defaults to 0.4
    :type pad: float, optional

    :return: Total number of threads to be launched
    :rtype: int
    """
    nthreads = int(psutil.cpu_count(logical=False) * pad)
    return nthreads


def run_parallel(function, files, nthreads):
    """Calls ``function`` for each element of the iterator ``files``.
    ``nthreads`` sets the number of worker processes spawned at a time.

    :param function: The function that is to be executed over each
        element of the iterator
    :type function: function

    :param files: The iterator that contains the data to be looped over
    :type files: iterator object

    :param nthreads: Number of worker processes to be spawned at a time.
    :type nthreads: int
    """
    pool = mp.Pool(nthreads)
    pool.map_async(function, files).get(720000)
    pool.close()
    pool.join()
