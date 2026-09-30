from turbos.moments.pressure import dev_P
from turbos.gradients.pderiv import vec_grad
from turbos.gradients.pderiv import strain_tensor
import xarray as xr




def pressure_strain(P, u):
    Pi = dev_P(
        P
    )

    D = strain_tensor(
        u
    )

    return xr.Dataset(
        {
            "PiD": -(Pi * D).sum(("i","j")),
        }
    )