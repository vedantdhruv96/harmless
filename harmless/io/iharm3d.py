from harmless.io.base import BaseDump

__all__ = ["Iharm3dDump"]


class Iharm3dDump(BaseDump):
    """Reader for iharm3D HDF5 fluid dump files.

    .. note::
        Not yet implemented.
    """

    def __init__(self, fname, extras=None):
        """Constructor method for Iharm3dDump.

        :param fname: Path to the iharm3D dump file
        :type fname: str

        :param extras: List of extra quantity keys to read, defaults to None
        :type extras: list, optional

        :raises NotImplementedError: This reader is not yet implemented.
        """
        raise NotImplementedError("iharm3D reader is not yet implemented.")
