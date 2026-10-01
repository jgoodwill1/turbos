import numpy as np
import xarray as xr


def vector(
    x_comp,
    y_comp,
    z_comp,
    *,
    name=None,
) -> xr.DataArray:
    values = np.stack(
        [
            x_comp.values,
            y_comp.values,
            z_comp.values,
        ],
        axis=-1,
    )

    return xr.DataArray(
        values,
        dims=("x", "y", "c"),
        coords={
            "x": x_comp["x"],
            "y": x_comp["y"],
            "c": ["x", "y", "z"],
        },
        name=name,
    )