import xarray as xr
import numpy as np


def vec_grad(v, spatial_dims=("x", "y")):

    if "c" not in v.dims:
        raise ValueError(
            f"Expected vector with 'c' dimension. Got {v.dims}"
        )

    gradients = []

    for dim in ("x", "y", "z"):

        if dim in spatial_dims:
            dv = v.differentiate(dim)
        else:
            dv = xr.zeros_like(v)

        gradients.append(
            dv.expand_dims(j=[dim])
        )

    grad = xr.concat(
        gradients,
        dim="j",
    )

    grad = grad.rename(
        {"c": "i"}
    )

    # Preserve the original non-component dimension order
    base_dims = [
        d for d in v.dims
        if d != "c"
    ]

    return grad.transpose(
        *base_dims,
        "i",
        "j",
    )


def div(v, spatial_dims=("x", "y")):

    if "c" not in v.dims:
        raise ValueError(
            f"Expected vector with 'c' dimension. Got {v.dims}"
        )

    divergence = 0
    for dim in ("x", "y", "z"):
        if dim in spatial_dims:
            divergence += v.differentiate(dim).sel(c=dim)

    return divergence


def strain_tensor(u, spatial_dims=("x", "y")):
    """
    Compute the 3x3 traceless strain tensor D_ij.

    Parameters
    ----------
    u : xr.DataArray
        Vector field with dimensions such as
        ('x', 'y', 'c'),
        where c = ['x', 'y', 'z'].

    spatial_dims : tuple
        Spatial coordinates along which derivatives
        are calculated.

        For a 2D simulation:
            ('x', 'y')

        Derivatives with respect to z are assumed zero.

    Returns
    -------
    xr.DataArray
        Traceless strain tensor with dimensions
        (..., i, j),

        where
            i = ['x', 'y', 'z']
            j = ['x', 'y', 'z']
    """

    # Full 3x3 velocity-gradient tensor
    grad_u = vec_grad(
        u,
        spatial_dims=spatial_dims,
    )

    # True tensor transpose: (grad u)^T
    grad_u_T = xr.DataArray(
        grad_u.data.swapaxes(-2, -1),
        dims=grad_u.dims,
        coords=grad_u.coords,
    )

    # Symmetric strain-rate tensor
    S = 0.5 * (
        grad_u + grad_u_T
    )

    # div(u) = du_x/dx + du_y/dy + du_z/dz
    divergence = (
        grad_u.sel(i="x", j="x")
        + grad_u.sel(i="y", j="y")
        + grad_u.sel(i="z", j="z")
    )

    # 3x3 identity tensor
    I = xr.DataArray(
        np.eye(3),
        dims=("i", "j"),
        coords={
            "i": ["x", "y", "z"],
            "j": ["x", "y", "z"],
        },
    )

    # Traceless strain tensor
    D = (
        S
        - (divergence / 3.0) * I
    )

    return D.transpose(
        *[d for d in u.dims if d != "c"],
        "i",
        "j",
    )