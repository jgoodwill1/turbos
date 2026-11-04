import numpy as np
import xarray as xr


def P_tensor(
    xx,
    xy,
    xz,
    yy,
    yz,
    zz,
    name=None,
):
    """
    Build a symmetric 3x3 tensor from six independent components.

    Inputs are xarray DataArrays with identical sample dimensions.
    """

    sample_dims = xx.dims
    sample_coords = {
        dim: xx.coords[dim]
        for dim in sample_dims
    }

    shape = xx.shape + (3, 3)

    values = np.empty(
        shape,
        dtype=np.result_type(
            xx.dtype,
            xy.dtype,
            xz.dtype,
            yy.dtype,
            yz.dtype,
            zz.dtype,
        ),
    )

    values[..., 0, 0] = xx
    values[..., 0, 1] = xy
    values[..., 0, 2] = xz

    values[..., 1, 0] = xy
    values[..., 1, 1] = yy
    values[..., 1, 2] = yz

    values[..., 2, 0] = xz
    values[..., 2, 1] = yz
    values[..., 2, 2] = zz

    return xr.DataArray(
        values,
        dims=sample_dims + ("i", "j"),
        coords={
            **sample_coords,
            "i": ["x", "y", "z"],
            "j": ["x", "y", "z"],
        },
        name=name,
    )


def scalar_pressure(P):
    return (
        P.sel(i="x", j="x")
        + P.sel(i="y", j="y")
        + P.sel(i="z", j="z")
    ) / 3

def dev_P(P):
    p = scalar_pressure(P)

    I = xr.DataArray(
        np.eye(3),
        dims=("i", "j"),
        coords={
            "i": ["x", "y", "z"],
            "j": ["x", "y", "z"],
        },
    )

    return P - p * I