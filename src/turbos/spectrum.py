import numpy as np
import xarray as xr
from scipy.fft import fftn, fftfreq


def energy_spectrum(
    da,
    spatial_dims=("x", "y"),
    component_dim="c",
    subtract_mean=True,
    energy_factor=0.5,
):
    """
    Compute the isotropic 2D energy spectrum E(k).

    Parameters
    ----------
    da : xr.DataArray
        Scalar or vector field.

        Examples:
            scalar: (x, y)
            vector: (x, y, c)

    spatial_dims : tuple
        Spatial dimensions over which to Fourier transform.

    component_dim : str
        Vector-component dimension. Ignored if absent.

    subtract_mean : bool
        Remove the spatial mean before computing the spectrum.

    energy_factor : float
        Multiplicative factor in the energy definition.

        Default:
            E = 1/2 |f|^2

    Returns
    -------
    xr.DataArray
        Isotropic shell-integrated energy spectrum E(k).

        The normalization is chosen so that approximately

            integral E(k) dk
                =
            energy_factor * <|f|^2>

        for the resolved field.
    """

    # ---------------------------------------------------------
    # Validate grid
    # ---------------------------------------------------------

    if len(spatial_dims) != 2:
        raise ValueError(
            "energy_spectrum currently expects two spatial dimensions."
        )

    for dim in spatial_dims:
        if dim not in da.dims:
            raise ValueError(
                f"{dim!r} is not present in {da.dims}"
            )

    dx = float(np.mean(np.diff(da[spatial_dims[0]].values)))
    dy = float(np.mean(np.diff(da[spatial_dims[1]].values)))

    nx = da.sizes[spatial_dims[0]]
    ny = da.sizes[spatial_dims[1]]

    # ---------------------------------------------------------
    # Remove spatial mean
    # ---------------------------------------------------------

    field = da

    if subtract_mean:
        field = field - field.mean(spatial_dims)

    # ---------------------------------------------------------
    # Fourier transform
    # ---------------------------------------------------------

    axes = tuple(
        field.get_axis_num(dim)
        for dim in spatial_dims
    )

    fhat = fftn(
        field.values,
        axes=axes,
    )

    # ---------------------------------------------------------
    # Modal energy
    #
    # Parseval:
    #
    # mean(|f|^2)
    #     =
    # sum(|F|^2) / (Nx Ny)^2
    # ---------------------------------------------------------

    normalization = (nx * ny) ** 2

    modal_energy = (
        energy_factor
        * np.abs(fhat) ** 2
        / normalization
    )

    # If vector-valued, sum over vector components.
    if component_dim in field.dims:
        component_axis = field.get_axis_num(component_dim)

        modal_energy = modal_energy.sum(
            axis=component_axis
        )

    # ---------------------------------------------------------
    # Wavenumbers
    # ---------------------------------------------------------

    kx = 2.0 * np.pi * fftfreq(
        nx,
        d=dx,
    )

    ky = 2.0 * np.pi * fftfreq(
        ny,
        d=dy,
    )

    KX, KY = np.meshgrid(
        kx,
        ky,
        indexing="ij",
    )

    kmag = np.sqrt(
        KX**2 + KY**2
    )

    # ---------------------------------------------------------
    # Isotropic shells
    # ---------------------------------------------------------

    dkx = 2.0 * np.pi / (nx * dx)
    dky = 2.0 * np.pi / (ny * dy)

    dk = min(dkx, dky)

    kmax = kmag.max()

    edges = np.arange(
        0.0,
        kmax + dk,
        dk,
    )

    shell_energy, _ = np.histogram(
        kmag.ravel(),
        bins=edges,
        weights=modal_energy.ravel(),
    )

    # Spectrum density so sum(E(k) dk) = total energy
    E = shell_energy / dk

    k = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    # ---------------------------------------------------------
    # Return xarray
    # ---------------------------------------------------------

    spectrum = xr.DataArray(
        E,
        dims=("k",),
        coords={"k": k},
        name="E",
    )

    spectrum.attrs.update({
        "long_name": "isotropic energy spectrum",
        "spatial_dims": spatial_dims,
        "subtract_mean": subtract_mean,
        "energy_factor": energy_factor,
    })

    return spectrum