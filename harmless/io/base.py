from harmless import diagnostics

__all__ = ["BaseDump"]


class BaseDump:
    """Abstract base class for GRMHD fluid dump readers.

    Implements lazy evaluation: derived quantities are computed on first
    access and cached in ``self.cache``. Primitives and derived quantities
    are accessed uniformly via ``dump['key']``.

    Subclasses must populate the standard primitives (``rho``, ``u``,
    ``u1``–``u3``, ``B1``–``B3``) as instance attributes in ``__init__``.

    A :class:`harmless.grid.Grid` must be attached via :meth:`set_grid`
    before requesting any metric-dependent quantity (``ucon``, ``bcon``,
    ``bsq``, stress-energy tensors, flux profiles, etc.).

    Example::

        dump = KHARMADump("torus.out0.00000.h5")
        G = Grid(...)
        dump.set_grid(G)

        bsq  = dump['bsq']          # computed and cached
        beta = dump['beta']         # reuses cached bsq -- free
        pg   = dump['pg']           # no grid needed
    """

    def set_grid(self, G):
        """Attach a :class:`harmless.grid.Grid` to the dump.

        Required before requesting any metric-dependent quantity.
        Clears the cache so stale values are not returned if the grid changes.

        :param G: Grid object
        :type G: :class:`harmless.grid.Grid`
        """
        self.G = G
        self.cache = {}

    def __getitem__(self, key):
        """Retrieve a primitive or derived quantity by string key.

        Resolution order:

        1. **Cache** – return immediately if already computed.
        2. **Diagnostic dict** – compute via
           :data:`harmless.diagnostics.diagnostic_dict`, cache, and return.
        3. **Primitive attribute** – return ``getattr(self, key)`` as a fallback
           for raw primitives set by the reader (``rho``, ``u``, ``B1``, etc.).

        :param key: Variable name (e.g. ``'rho'``, ``'pg'``, ``'ucon'``)
        :type key: str

        :return: The requested array
        :raises KeyError: If the key is not a primitive attribute or a known
            diagnostic key.
        """
        if not hasattr(self, "cache"):
            self.cache = {}

        if key in self.cache:
            return self.cache[key]

        if key in diagnostics.diagnostic_dict:
            self.cache[key] = diagnostics.diagnostic_dict[key](self)
            return self.cache[key]

        if hasattr(self, key):
            return getattr(self, key)

        raise KeyError(
            f"'{key}' is not a primitive attribute or a known diagnostic key."
        )

    def get_derived(self, var, G=None, components=None):
        """Compute or retrieve a derived quantity, using the cache.

        Wraps :meth:`__getitem__` with optional grid attachment and
        component selection for tensor quantities.

        :param var: Variable key (e.g. ``'pg'``, ``'ucon'``, ``'Tmixed'``)
        :type var: str

        :param G: Grid object; stored on the dump if provided
        :type G: :class:`harmless.grid.Grid`, optional

        :param components: Tensor component ``(mu, nu)`` to extract, or
            ``None`` for the full array
        :type components: tuple, optional

        :return: Computed quantity (full array or a single component)
        :raises KeyError: If var is not a recognised diagnostic key
        """
        if G is not None:
            self.G = G
            if not hasattr(self, "cache"):
                self.cache = {}

        result = self[var]

        if components is not None:
            mu, nu = components
            return result[..., mu, nu]
        return result
