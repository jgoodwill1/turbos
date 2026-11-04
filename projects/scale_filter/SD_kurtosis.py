import numpy as np
import xarray as xr


def increment_kurtosis(f, dim, shifts):
    """Spatial increment kurtosis versus integer grid lag."""
    h = abs(float(f[dim].values[1] - f[dim].values[0]))
    values = []

    for shift in shifts:
        delta = (
            f.roll({dim: -int(shift)}, roll_coords=False) - f
        )

        m2 = (delta**2).mean(("x", "y"), skipna=True)
        m4 = (delta**4).mean(("x", "y"), skipna=True)
        values.append(m4 / m2.where(m2 > 0)**2)

    return xr.concat(
        values,
        dim=xr.IndexVariable("ell", np.asarray(shifts) * h),
    )