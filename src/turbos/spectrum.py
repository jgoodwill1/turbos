
import numpy as np
import xarray as xr


def energy_spectrum(ar, sumax=2, lenx=2*np.pi, leny=2*np.pi, lenz=2*np.pi):
    """
    Compute the perpendicular energy spectrum of a scalar or vector field.

    Parameters
    ----------
    ar : xr.DataArray
        Scalar field: dimensions ('x', 'y')
        Vector field: dimensions ('x', 'y', 'c')
    sumax : int
        Magnetic-field direction (only sumax=2 supported).
    lenx, leny, lenz : float
        Physical domain lengths.

    Returns
    -------
    kk : xr.DataArray
        Perpendicular wavenumbers.
    fekp : xr.DataArray
        Perpendicular energy spectrum.
    """

    if sumax != 2:
        raise NotImplementedError("Only sumax=2 is supported.")

    # Detect scalar or vector field
    vector = "c" in ar.dims

    if vector:
        ar = ar.transpose("x", "y", "c")
    else:
        ar = ar.transpose("x", "y")

    arr = ar.to_numpy()
    nx, ny = arr.shape[:2]

    if nx < 2 or ny < 2:
        raise ValueError("Both spatial dimensions must contain at least 2 points.")

    # Remove spatial mean independently for each component
    arr = arr - np.mean(arr, axis=(0, 1), keepdims=True)

    # Original wavenumber calculation
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx * (2*np.pi/lenx)
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny * (2*np.pi/leny)

    # FFT along spatial dimensions only
    far = np.fft.fftshift(
        np.fft.fftn(arr, axes=(0, 1)),
        axes=(0, 1)
    ) / (nx * ny)

    # Fourier energy
    if vector:
        energy = 0.5 * np.sum(np.abs(far)**2, axis=2)
    else:
        energy = 0.5 * np.abs(far)**2

    # Perpendicular wavenumber magnitude
    kp = np.sqrt(kx[:, None]**2 + ky[None, :]**2)

    # Original radial-bin definitions
    dk = np.abs(kp[1, 0] - kp[0, 0])
    kk = kp[nx//2, ny//2:]

    nbins = min(nx//2, ny//2)
    kk = kk[:nbins]

    # Sort wavenumbers and corresponding Fourier energies
    order = np.argsort(kp.ravel())
    k_sorted = kp.ravel()[order]
    e_sorted = energy.ravel()[order]

    # Preserve original strict radial-shell boundaries
    lower = np.searchsorted(k_sorted, kk - dk, side="right")
    upper = np.searchsorted(k_sorted, kk + dk, side="left")

    # Direct shell summation (avoids cumulative-sum cancellation)
    fekp = np.array([
        np.sum(e_sorted[lo:hi])
        for lo, hi in zip(lower, upper)
    ]) / dk

    # Return xarray objects
    kk = xr.DataArray(
        kk,
        dims="kperp",
        name="kperp"
    )

    fekp = xr.DataArray(
        fekp,
        dims="kperp",
        coords={"kperp": kk.values},
        name="E_kperp"
    )

    return kk, fekp
