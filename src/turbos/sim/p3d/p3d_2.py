"""
Minimal P3D reader.

Goals
-----
- Support legacy P3D movie files and newer HDF5 snapshots.
- Return a consistent xarray.Dataset.
- Keep file-format logic separate from scientific analysis.
- Support canonical variables such as B, E, ue, ui, Pe, Pi.
"""

from pathlib import Path

import numpy as np
import xarray as xr


# Native fields that can be read directly from P3D output.
PRIMITIVES = {
    "bx", "by", "bz",
    "ex", "ey", "ez",
    "jx", "jy", "jz",
    "jex", "jey", "jez",
    "jix", "jiy", "jiz",
    "ne", "ni", "rho",
    "pexx", "pexy", "pexz", "peyy", "peyz", "pezz",
    "pixx", "pixy", "pixz", "piyy", "piyz", "pizz",
}


# Canonical variables exposed to downstream turbos analysis.
# These map only to native P3D fields; derived assembly happens after loading.
CANONICAL_DEPENDENCIES = {
    "B":  ("bx", "by", "bz"),
    "E":  ("ex", "ey", "ez"),
    "J":  ("jx", "jy", "jz"),
    "Je": ("jex", "jey", "jez"),
    "Ji": ("jix", "jiy", "jiz"),

    # Velocity is derived directly from current and density.
    "ue": ("jex", "jey", "jez", "ne"),
    "ui": ("jix", "jiy", "jiz", "ni"),

    "Pe": ("pexx", "pexy", "pexz", "peyy", "peyz", "pezz"),
    "Pi": ("pixx", "pixy", "pixz", "piyy", "piyz", "pizz"),
}

CANONICAL_ALL = [
    "B", "E", "J", "Je", "Ji", "ue", "ui", "Pe", "Pi", "ne", "ni", "rho",
]


class P3D:
    """
    Small reader for P3D simulation output.

    Parameters
    ----------
    rundir : str or Path
        P3D run directory.
    filenum : str, default "000"
        Legacy movie-file number.
    """

    def __init__(self, rundir, filenum="000"):
        self.rundir = Path(rundir).expanduser().resolve()
        self.name = self.rundir.name
        self.filenum = str(filenum).zfill(3)

        self.paramfile = self._find_paramfile()
        self.params = load_params(self.paramfile)

        # Make common run parameters available as attributes.
        for key, value in self.params.items():
            setattr(self, key, value)

        self.format = self._detect_format()

        # Loaded lazily only for compressed legacy movie files.
        self._movie_log = None
        self.numslices = self._get_num_slices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_xarray(self, slices, variables):
        """
        Load one or more snapshots as a canonical xarray.Dataset.

        Examples
        --------
        ds = run.load_xarray(
            0,
            variables=["B", "ue", "Pe", "ne"],
        )

        ds = run.load_xarray(
            range(10),
            variables=["B", "Pe"],
        )
        """
        if variables == "all":
            variables = CANONICAL_ALL
        elif variables == "all_raw":
            variables = sorted(PRIMITIVES)
        elif isinstance(variables, str):
            variables = [variables]

        requested = list(variables)
        native = self._resolve_native(requested)

        if isinstance(slices, (int, np.integer)):
            return self._load_one(int(slices), requested, native)

        datasets = [
            self._load_one(int(index), requested, native)
            for index in slices
        ]

        if not datasets:
            raise ValueError("slices must contain at least one snapshot")

        return xr.concat(
            [
                ds.expand_dims(time=[ds.attrs["time"]])
                for ds in datasets
            ],
            dim="time",
        )

    # Optional shorter alias.
    snapshot = load_xarray

    # ------------------------------------------------------------------
    # Format-independent snapshot path
    # ------------------------------------------------------------------

    def _load_one(self, index, requested, native):
        if self.format == "hdf5":
            raw = self._load_hdf5(index, native)
        else:
            raw = self._load_movie(index, native)

        ds = self._canonicalize(raw, requested)

        ds.attrs.update({
            "run": self.name,
            "snapshot": index,
            "format": self.format,
        })

        return ds

    def _resolve_native(self, requested):
        native = []

        for name in requested:
            deps = CANONICAL_DEPENDENCIES.get(name, (name,))

            for dep in deps:
                if dep not in PRIMITIVES:
                    raise ValueError(
                        f"Unknown variable {name!r}; "
                        f"{dep!r} is not a known native P3D field."
                    )

                if dep not in native:
                    native.append(dep)

        return native

    # ------------------------------------------------------------------
    # Legacy movie files
    # ------------------------------------------------------------------

    def _load_movie(self, index, variables):
        coords = self._coords()
        shape = (self.nx, self.ny, self.nz)

        data = {}

        for name in variables:
            path = self._movie_path(name)
            array = self._read_movie_field(path, name, index)

            if array.shape != shape:
                raise ValueError(
                    f"{name!r} has shape {array.shape}; expected {shape}"
                )

            data[name] = (("x", "y", "z"), array)

        ds = xr.Dataset(data, coords=coords)
        ds.attrs["time"] = index * self.dtmovie

        return ds.squeeze(drop=True)

    def _movie_path(self, name):
        direct = self.rundir / name

        if direct.is_file():
            return direct

        staged = (
            self.rundir
            / "staging"
            / f"movie.{name}.{self.filenum}"
        )

        if staged.is_file():
            return staged

        raise FileNotFoundError(
            f"Could not find legacy movie field {name!r}"
        )

    def _read_movie_field(self, path, name, index):
        n = self.nx * self.ny * self.nz
        shape = (self.nx, self.ny, self.nz)

        dtype_info = {
            "b":  (np.uint8,   1),
            "bb": (np.int16,   2),
            "f":  (np.float32, 4),
            "d":  (np.float64, 8),
        }

        if self.data_type not in dtype_info:
            raise ValueError(
                f"Unsupported P3D data_type {self.data_type!r}"
            )

        dtype, bytes_per_value = dtype_info[self.data_type]
        offset = index * n * bytes_per_value

        with path.open("rb") as f:
            f.seek(offset)
            values = np.fromfile(f, dtype=dtype, count=n)

        if values.size != n:
            raise IndexError(
                f"Snapshot {index} is incomplete or absent in {path}"
            )

        values = values.reshape(shape, order="F")

        # Legacy byte-compressed movie data require min/max reconstruction.
        if self.data_type in ("b", "bb"):
            vmin, vmax = self._movie_minmax(name, index)

            if self.data_type == "b":
                values = vmin + (vmax - vmin) * values / 255.0
            else:
                # Preserve the normalization used by the legacy reader.
                values = (
                    vmin
                    + (vmax - vmin)
                    * (values + 32678.0)
                    / 65535.0
                )

        return values.astype(np.float64, copy=False)

    def _movie_minmax(self, name, index):
        if self._movie_log is None:
            log_path = self.rundir / "log"

            if not log_path.is_file():
                log_path = (
                    self.rundir
                    / "staging"
                    / f"movie.log.{self.filenum}"
                )

            if not log_path.is_file():
                raise FileNotFoundError(
                    "Compressed movie data require a movie log file."
                )

            self._movie_log = np.loadtxt(log_path)

        if name not in self.logvars:
            raise KeyError(
                f"{name!r} is not present in movie log variable list"
            )

        row = index * len(self.logvars) + self.logvars.index(name)
        return self._movie_log[row]

    # ------------------------------------------------------------------
    # HDF5 snapshots
    # ------------------------------------------------------------------

    def _load_hdf5(self, index, variables):
        import h5py

        path = self.rundir / f"{self._hdf_prefix}.{index:03d}.dat"

        if not path.is_file():
            raise FileNotFoundError(
                f"HDF5 snapshot not found: {path}"
            )

        with h5py.File(path, "r") as h5:
            coords = {
                "x": np.asarray(h5["xx"][()]),
                "y": np.asarray(h5["yy"][()]),
                "z": np.asarray(h5["zz"][()]),
            }

            shape = (self.nx, self.ny, self.nz)
            data = {}

            for name in variables:
                if name not in h5:
                    raise KeyError(
                        f"{name!r} not found in {path.name}"
                    )

                array = h5[name][...]

                if array.shape != shape:
                    raise ValueError(
                        f"{name!r} has shape {array.shape}; "
                        f"expected {shape}"
                    )

                data[name] = (("x", "y", "z"), array)

            time = float(h5["time"][()])

        ds = xr.Dataset(data, coords=coords)
        ds.attrs["time"] = time

        return ds.squeeze(drop=True)

    # ------------------------------------------------------------------
    # Canonical xarray representation
    # ------------------------------------------------------------------

    def _canonicalize(self, raw, requested):
        out = xr.Dataset(
            attrs=raw.attrs.copy()
        )

        for name in requested:
            if name in raw:
                out[name] = raw[name]

            elif name in ("B", "E", "J", "Je", "Ji"):
                out[name] = _vector(
                    raw,
                    CANONICAL_DEPENDENCIES[name],
                    name,
                )

            elif name == "ue":
                out[name] = _vector_from_arrays(
                    -raw["jex"] / raw["ne"],
                    -raw["jey"] / raw["ne"],
                    -raw["jez"] / raw["ne"],
                    name,
                )

            elif name == "ui":
                out[name] = _vector_from_arrays(
                    raw["jix"] / raw["ni"],
                    raw["jiy"] / raw["ni"],
                    raw["jiz"] / raw["ni"],
                    name,
                )

            elif name in ("Pe", "Pi"):
                out[name] = _symmetric_tensor(
                    raw,
                    CANONICAL_DEPENDENCIES[name],
                    name,
                )

            else:
                raise KeyError(
                    f"Could not construct requested variable {name!r}"
                )

        return out

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _find_paramfile(self):
        candidates = [
            self.rundir / f"param_{self.name}",
            self.rundir / "staging" / f"param_{self.name}",
            self.rundir / "paramfile",
        ]

        for path in candidates:
            if path.is_file():
                return path

        raise FileNotFoundError(
            f"No P3D parameter file found in {self.rundir}"
        )

    def _detect_format(self):
        snapshots = sorted(self.rundir.glob("*.???.dat"))

        for path in snapshots:
            with path.open("rb") as f:
                if f.read(8) == b"\x89HDF\r\n\x1a\n":
                    self._hdf_prefix = path.name.rsplit(".", 2)[0]
                    return "hdf5"

        return "movie"

    def _coords(self):
        return {
            "x": np.arange(self.nx) * self.dx,
            "y": np.arange(self.ny) * self.dy,
            "z": np.arange(self.nz) * self.dz,
        }
    
    def _get_num_slices(self):
        if self.format == "hdf5":
            return len(
                list(self.rundir.glob(f"{self._hdf_prefix}.???.dat"))
            )

        # legacy movie files
        sample = self._movie_path("bx")

        bytes_per_value = {
            "b": 1,
            "bb": 2,
            "f": 4,
            "d": 8,
        }[self.data_type]

        bytes_per_slice = (
            self.nx
            * self.ny
            * self.nz
            * bytes_per_value
        )

        return sample.stat().st_size // bytes_per_slice


# ----------------------------------------------------------------------
# Canonical field builders
# ----------------------------------------------------------------------

def _vector(ds, names, name):
    return _vector_from_arrays(
        ds[names[0]],
        ds[names[1]],
        ds[names[2]],
        name,
    )


def _vector_from_arrays(x, y, z, name):
    spatial_dims = x.dims

    da = xr.concat(
        [x, y, z],
        dim=xr.IndexVariable(
            "c",
            ["x", "y", "z"],
        ),
    )

    return da.transpose(
        *spatial_dims,
        "c",
    ).rename(name)


def _symmetric_tensor(ds, names, name):
    xx, xy, xz, yy, yz, zz = [
        ds[n] for n in names
    ]

    labels = ["x", "y", "z"]

    row_x = xr.concat(
        [xx, xy, xz],
        dim=xr.IndexVariable("j", labels),
    )

    row_y = xr.concat(
        [xy, yy, yz],
        dim=xr.IndexVariable("j", labels),
    )

    row_z = xr.concat(
        [xz, yz, zz],
        dim=xr.IndexVariable("j", labels),
    )

    tensor = xr.concat(
        [row_x, row_y, row_z],
        dim=xr.IndexVariable("i", labels),
    )

    return tensor.transpose(
        *xx.dims,
        "i",
        "j",
    ).rename(name)


# ----------------------------------------------------------------------
# Parameter file
# ----------------------------------------------------------------------

def load_params(paramfile):
    params = {}

    with open(paramfile) as f:
        for line in f:
            parts = line.split()

            if (
                len(parts) < 2
                or parts[0] != "#define"
            ):
                continue

            key = parts[1]

            if len(parts) == 2:
                value = True
            else:
                raw = parts[2]

                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw

            params[key] = value

    # Data type.
    if "eight_byte" in params:
        params["data_type"] = "d"
    elif "four_byte" in params:
        params["data_type"] = "f"
    elif "double_byte" in params:
        params["data_type"] = "bb"
    else:
        params["data_type"] = "b"

    # Movie cadence.
    if "n_movieout" in params:
        params["dtmovie"] = (
            params["n_movieout"] * params["dt"]
        )
    else:
        params["dtmovie"] = params["movieout"]

    # Global grid.
    params["nx"] = int(params["pex"] * params["nx"])
    params["ny"] = int(params["pey"] * params["ny"])
    params["nz"] = int(params["pez"] * params["nz"])

    params["dx"] = params["lx"] / params["nx"]
    params["dy"] = params["ly"] / params["ny"]
    params["dz"] = params["lz"] / params["nz"]

    # Needed only for byte-compressed legacy movie output.
    movie_header = params.get("movie_header")

    movie_logvars = {
        '"movie2dC.h"': [
            "rho", "jx", "jy", "jz",
            "bx", "by", "bz",
            "ex", "ey", "ez",
            "ne", "jex", "jey", "jez",
            "pexx", "peyy", "pezz",
            "pexy", "peyz", "pexz",
            "ni",
            "pixx", "piyy", "pizz",
            "pixy", "piyz", "pixz",
        ],
        '"movie4b.h"': [
            "rho", "jx", "jy", "jz",
            "bx", "by", "bz",
            "ex", "ey", "ez",
            "ne", "jex", "jey", "jez",
            "pexx", "peyy", "pezz",
            "pexz", "peyz", "pexy",
            "ni", "jix", "jiy", "jiz",
            "pixx", "piyy", "pizz",
            "pixz", "piyz", "pixy",
        ],
        '"movie2dD.h"': [
            "rho", "jx", "jy", "jz",
            "bx", "by", "bz",
            "ex", "ey", "ez",
            "ne", "jex", "jey", "jez",
            "pexx", "peyy", "pezz",
            "pexy", "peyz", "pexz",
            "ni", "jix", "jiy", "jiz",
            "pixx", "piyy", "pizz",
            "pixy", "piyz", "pixz",
        ],
        '"movie_pic3.0.h"': [
            "rho", "jx", "jy", "jz",
            "bx", "by", "bz",
            "ex", "ey", "ez",
            "ne", "jex", "jey", "jez",
            "pexx", "peyy", "pezz",
            "pexy", "peyz", "pexz",
            "ni", "jix", "jiy", "jiz",
            "pixx", "piyy", "pizz",
            "pixy", "piyz", "pixz",
        ],
    }

    params["logvars"] = movie_logvars.get(
        movie_header,
        [],
    )

    return params
    
