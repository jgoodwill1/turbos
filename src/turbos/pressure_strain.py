from turbos.moments.pressure import dev_P, scalar_pressure
from turbos.gradients.pderiv import div, vec_grad, strain_tensor
import xarray as xr




def pressure_strain(P, u):
    
    p = scalar_pressure(
        P
    )

    th = div(u)
 
    Pi = dev_P(
        P
    )

    D = strain_tensor(
        u
    )


    return xr.Dataset(
        {
            "pth": -p * th,
            "PiD": -(Pi * D).sum(("i","j")),
        }
    )