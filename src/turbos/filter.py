import numpy as np
import xarray as xr
from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq


def kfilter(
    da,
    kcut,
    spatial_dims=("x", "y"),
):
    """
    Isotropic sharp low-pass Fourier filter.

    Parameters
    ----------
    da : xr.DataArray
        Input field. May be scalar, vector, or tensor, e.g.

            (x, y)
            (x, y, c)
            (x, y, i, j)

    kcut : float
        Wavenumber cutoff. Modes satisfying

            sqrt(kx**2 + ky**2) <= kcut

        are retained.

        If x and y are measured in d_i, then k is in 1/d_i,
        so kcut=13 corresponds to k_f d_i = 13.

    spatial_dims : tuple
        Dimensions over which to perform the Fourier transform.

    Returns
    -------
    xr.DataArray
        Filtered field with the same dimensions, coordinates,
        name, and attributes as the input.
    """

    # ---------------------------------------------------------
    # Validate spatial dimensions and obtain grid spacing
    # ---------------------------------------------------------

    spacings = {}

    for dim in spatial_dims:

        if dim not in da.dims:
            raise ValueError(
                f"{dim!r} is not a dimension of {da.dims}"
            )

        coord = np.asarray(da[dim].values)

        if coord.size < 2:
            raise ValueError(
                f"{dim!r} must contain at least two points."
            )

        delta = np.diff(coord)

        if not np.allclose(delta, delta[0]):
            raise ValueError(
                f"{dim!r} must be uniformly spaced."
            )

        spacings[dim] = float(delta[0])

    # Axes of x/y inside the full array.
    axes = tuple(
        da.get_axis_num(dim)
        for dim in spatial_dims
    )

    # ---------------------------------------------------------
    # Fourier transform
    # ---------------------------------------------------------

    values = np.asarray(da.values)

    fhat = rfftn(
        values,
        axes=axes,
    )

    # ---------------------------------------------------------
    # Construct wavenumber grid
    #
    # rfftn uses rfft only on the LAST transformed dimension.
    # ---------------------------------------------------------

    k_components = []

    for n, dim in enumerate(spatial_dims):

        N = da.sizes[dim]
        dx = spacings[dim]

        if n == len(spatial_dims) - 1:
            k = 2.0 * np.pi * rfftfreq(
                N,
                d=dx,
            )
        else:
            k = 2.0 * np.pi * fftfreq(
                N,
                d=dx,
            )

        shape = [1] * fhat.ndim
        shape[axes[n]] = k.size

        k_components.append(
            k.reshape(shape)
        )

    k2 = sum(
        k**2
        for k in k_components
    )

    # ---------------------------------------------------------
    # Sharp isotropic low-pass filter
    # ---------------------------------------------------------

    mask = k2 <= kcut**2

    fhat *= mask

    # ---------------------------------------------------------
    # Back transform
    # ---------------------------------------------------------

    filtered = irfftn(
        fhat,
        s=[
            da.sizes[dim]
            for dim in spatial_dims
        ],
        axes=axes,
    )

    # ---------------------------------------------------------
    # Restore xarray metadata
    # ---------------------------------------------------------

    out = xr.DataArray(
        filtered,
        dims=da.dims,
        coords=da.coords,
        name=da.name,
        attrs=da.attrs.copy(),
    )

    out.attrs["kfilter_cutoff"] = kcut

    return out