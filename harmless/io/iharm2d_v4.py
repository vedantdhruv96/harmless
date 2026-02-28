from harmless.io.base import BaseDump

__all__ = ["Iharm2dv4Dump"]


class Iharm2dv4Dump(BaseDump):
    """Reader for iharm2d_v4 HDF5 fluid dump files.

    .. note::
        Not yet implemented.
    """

    def __init__(self, fname, extras=None):
        """Constructor method for Iharm2dv4Dump.

        :param fname: Path to the iharm2d_v4 dump file
        :type fname: str

        :param extras: List of extra quantity keys to read, defaults to None
        :type extras: list, optional

        :raises NotImplementedError: This reader is not yet implemented.
        """
        raise NotImplementedError("iharm2d_v4 reader is not yet implemented.")
