"""Read assembled VPIC HDF5 snapshots into the same xarray layout as P3D.

Dependencies: numpy, xarray, h5py. Arrays are loaded eagerly and files closed.
This reader handles global arrays, not distributed rank-local dumps or halos.

Example
-------
run = VPIC(
    info_file='/run/info',
    data_dirs={'fields': '/data/fields',
               'hydro': '/data/hydro_hdf5'},
    axis_order=('x', 'y', 'z'),  # MUST match the HDF5 array storage order
    params={'dt': 0.01},        # override/add metadata as necessary
)
ds = run.load_xarray(0, variables=['B', 'E'])  # first available output
# run.load_step(40000, ['B'])                 # actual simulation step
# run.available_variables(0)                 # inspect native dataset names

Fields always use T.{step}/fields_{step}.h5 and contain
Timestep_{step}/cbx, cby, cbz, ex, ey, ez.
Both species share hydro_hdf5/T.{step}/, with files
hydro_electron_{step}.h5 and hydro_ion_{step}.h5. Pass their parent using
data_dirs={'fields': ..., 'hydro': ...}.
Customize file_patterns, group_pattern, and variable_map for your dump format:
    variable_map={'rhoe': ('electron', 'rho')}
Mapping keys are P3D-style native names; values are (folder key, dataset path).
Arbitrary scalar raw fields can be added with the same mapping mechanism.

Hydro rho is signed charge density: ne=rhoe/q_e, ni=rhoi/q_i.
Default q_e=-1 and q_i=+1 match normalized moments. Tiny qe/qi in the info
report may describe macroparticle weights and are not used as species charges.
Species velocities use J_species/rho_species; no mass is needed.
Te/Ti expose the uncentered stress tensors, using the stored tzx for xz.
Momentum_e/momentum_i expose px,py,pz; kee/kei expose kinetic energy density.
Pe/Pi are calculated automatically with m_e=1 by default and m_i
from the info-file mi/me ratio (or an explicit m_i). They use P=T-m*n*u*u; apply this approximation
only to nonrelativistic runs with mutually consistent units. Relativistic
rest-frame pressure reconstruction is not implemented. Raw stress stays
available regardless of pressure mode.
cbx/cby/cbz are divided by magnetic_scale (default 1). Set magnetic_scale to
c if these datasets store c*B and you require B; the reader cannot infer units.
When wci is present, time is omega_ci*t; time_simulation retains raw time.
Without wci, time remains in simulation units. Coordinates start at zero.
Physical coordinates need dx/dy/dz or lx/ly/lz in the info file/params. Otherwise
coordinates are explicitly marked as cell indices. Origins default to zero;
coordinate_offset=0.5 selects cell centers instead of the default offset 0.
"""

from pathlib import Path
import json
import re
import numpy as np
import xarray as xr
import h5py

AXES = ('x', 'y', 'z')
PAIRS = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')
DEPENDENCIES = {
    'B': ('bx', 'by', 'bz'), 'E': ('ex', 'ey', 'ez'),
    'J': ('jex', 'jey', 'jez', 'jix', 'jiy', 'jiz'),
    'Je': ('jex', 'jey', 'jez'), 'Ji': ('jix', 'jiy', 'jiz'),
    'ne': ('rhoe',), 'ni': ('rhoi',), 'rho': ('rhoe', 'rhoi'),
    'ue': ('jex', 'jey', 'jez', 'rhoe'),
    'ui': ('jix', 'jiy', 'jiz', 'rhoi'),
    'Te': tuple('te' + p for p in PAIRS),
    'Ti': tuple('ti' + p for p in PAIRS),
    'momentum_e': ('pex', 'pey', 'pez'),
    'momentum_i': ('pix', 'piy', 'piz'),
}
for _species in ('e', 'i'):
    DEPENDENCIES['P' + _species] = (
        DEPENDENCIES['T' + _species] + DEPENDENCIES['u' + _species]
    )


def _default_map():
    result = {a + d: ('fields', ('c' if a == 'b' else '') + a + d)
              for a in ('b', 'e') for d in AXES}
    for suffix, folder in [('e', 'electron'), ('i', 'ion')]:
        result['rho' + suffix] = (folder, 'rho')
        result['ke' + suffix] = (folder, 'ke')
        result.update({'j' + suffix + d: (folder, 'j' + d) for d in AXES})
        result.update({'p' + suffix + d: (folder, 'p' + d) for d in AXES})
        result.update({'t' + suffix + pair: (folder, 't' + ('zx' if pair == 'xz' else pair))
                       for pair in PAIRS})
    return result


def load_info(path):
    """Read scalar metadata from JSON, HDF5 or simple text; never execute code.

    Text accepts '#define nx 1200', 'nx = 1200;', 'nx: 1200', or 'nx 1200'.
    VPIC report labels such as mi/me, Lx/de and dt*wpe are preserved.
    Duplicate labels retain their last value and are recorded in
    _info_duplicates. Numeric literals only; expressions and input decks are not
    evaluated. Use params={...} for unresolved values. HDF5 reads scalar datasets
    and root attributes. Nested JSON/HDF5 names are retained as written.
    """
    path = Path(path).expanduser().resolve()
    if h5py.is_hdf5(path):
        result = {}
        with h5py.File(path, 'r') as f:
            result.update({k: _scalar(v) for k, v in f.attrs.items()
                           if np.asarray(v).size == 1})
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.size == 1:
                    result[name] = _scalar(obj[()])
            f.visititems(visit)
        return result
    text = path.read_text()
    if path.suffix.lower() == '.json' or text.lstrip().startswith('{'):
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError('Info JSON must contain an object')
        return result
    result = {}
    duplicates = {}
    for line in text.splitlines():
        line = line.split('//', 1)[0].strip()
        if not line or line.startswith('***'):
            continue
        if line.startswith('#define '):
            parts = line[len('#define '):].split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        elif line.startswith('#'):
            continue
        elif '=' in line:
            key, value = line.split('=', 1)
        elif ':' in line:
            key, value = line.split(':', 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2 or not re.fullmatch(r'[A-Za-z_]\w*', parts[0]):
                continue
            key, value = parts
        key = key.strip()
        value = value.split('#', 1)[0].rstrip(';').strip()
        if not key or not value:
            continue
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value.replace('D', 'e').replace('d', 'e'))
            except ValueError:
                parsed = value.strip('"\'')
        if key in result:
            duplicates.setdefault(key, [result[key]]).append(parsed)
        result[key] = parsed
    if duplicates:
        result['_info_duplicates'] = duplicates
    if not result:
        raise ValueError(f'No metadata parsed from {path}; supply a scalar info file')
    return result


def _normalize_info(params):
    """Add reader aliases while preserving every original report label.

    For this report, lengths/spacings are in electron inertial length units.
    Coordinates start at zero by default; xmin/ymin/zmin are retained only
    as report metadata. Rank-local or duplicate xmax values never determine
    the global grid. The reader uses the specified electron-mass normalization m_e=1.
    """
    result = dict(params)
    for axis in AXES:
        count = result.get('n' + axis)
        if count is not None:
            count = float(count)
            if not np.isfinite(count) or count <= 0 or not count.is_integer():
                raise ValueError(f'n{axis} must be a positive integer')
            result['n' + axis] = int(count)
        spacing = 'd' + axis
        if spacing not in result:
            for candidate in ('h' + axis, spacing + '/de'):
                if candidate in result:
                    result[spacing] = float(result[candidate])
                    break
        length = 'l' + axis
        if length not in result and 'L' + axis + '/de' in result:
            result[length] = float(result['L' + axis + '/de'])
    if 'dt' not in result:
        for frequency in ('wpe', 'wce', 'wci', 'wpi'):
            product = 'dt*' + frequency
            if product in result and frequency in result:
                rate = float(result[frequency])
                if not np.isfinite(rate) or rate <= 0:
                    raise ValueError(f'{frequency} must be finite and positive')
                result['dt'] = float(result[product]) / rate
                break
    for canonical, alias in [('m_e', 'me'), ('m_i', 'mi')]:
        if canonical not in result and alias in result:
            result[canonical] = float(result[alias])
    return result


def _scalar(value):
    value = np.asarray(value).item()
    return value.decode() if isinstance(value, bytes) else value


class VPIC:
    """VPIC global HDF5 reader with independent info and data paths.

    data_dirs may be one fields-directory path or a mapping of folder keys to
    paths. A 'hydro' folder key supplies the shared electron/ion directory.
    Fields always use 'T.{step}/fields_{step}.h5'. file_patterns
    may customize the other folders. Discovery scans T.<step> directories
    containing the expected field file; indices are zero
    based positions in sorted available steps, NOT simulation step numbers.

    axis_order is the on-disk order, e.g. ('z','y','x') or ('y','x') for 2D.
    Singleton spatial axes are dropped by default, as in the supplied P3D reader.
    q_e/q_i default to normalized species charges -1/+1. Report qe/qi
    values are retained as metadata and never used automatically as species
    charges. Override q_e/q_i only for a different hydro-moment normalization.
    Electron mass defaults to 1; the reported mi/me ratio supplies ion mass.
    With n <= 0, derived velocities are NaN. Direct velocity components can be
    used with velocity_source='direct' and mappings for uex/uey/uez, etc.
    """

    def __init__(self, info_file, data_dirs, *, params=None,
                 file_patterns=None, group_pattern='Timestep_{step}',
                 variable_map=None, axis_order=('x', 'y', 'z'),
                 q_e=None, q_i=None, magnetic_scale=1.0,
                 velocity_source='current', m_e=1.0, m_i=None,
                 coordinate_offset=0.0,
                 squeeze=True):
        self.info_file = Path(info_file).expanduser().resolve()
        self.params = load_info(self.info_file)
        self.params.update(params or {})
        self.params = _normalize_info(self.params)
        q_e = float(self.params.get('q_e', -1.0) if q_e is None else q_e)
        q_i = float(self.params.get('q_i', 1.0) if q_i is None else q_i)
        if isinstance(data_dirs, (str, Path)):
            data_dirs = {'fields': data_dirs}
        self.data_dirs = {k: Path(v).expanduser().resolve()
                          for k, v in data_dirs.items()}
        if 'hydro' in self.data_dirs:
            hydro = self.data_dirs.pop('hydro')
            self.data_dirs.setdefault('electron', hydro)
            self.data_dirs.setdefault('ion', hydro)
        if 'fields' not in self.data_dirs:
            raise ValueError("data_dirs must include a 'fields' directory")
        self.file_patterns = {
            'fields': 'T.{step}/fields_{step}.h5',
            'electron': 'T.{step}/hydro_electron_{step}.h5',
            'ion': 'T.{step}/hydro_ion_{step}.h5',
        }
        self.file_patterns.update(file_patterns or {})
        self.file_patterns['fields'] = 'T.{step}/fields_{step}.h5'
        self.group_pattern = group_pattern
        self.variable_map = _default_map()
        self.variable_map.update(variable_map or {})
        for name, route in self.variable_map.items():
            if isinstance(route, str) or len(route) != 2:
                raise ValueError(f'Mapping for {name!r} must be (folder, dataset)')
        self.axis_order = tuple(axis_order)
        if not self.axis_order or len(set(self.axis_order)) != len(self.axis_order) or not set(self.axis_order) <= set(AXES):
            raise ValueError('axis_order must contain distinct axes from x, y, z')
        if velocity_source not in ('current', 'direct'):
            raise ValueError("velocity_source must be 'current' or 'direct'")
        if not all(np.isfinite(v) and v != 0 for v in (q_e, q_i, magnetic_scale)):
            raise ValueError('Charges and magnetic_scale must be finite and nonzero')
        self.masses = {
            'e': float(m_e) if m_e is not None else 1.0,
            'i': m_i if m_i is not None else self.params.get('m_i'),
        }
        ratio = self.params.get('mi/me')
        if ratio is not None:
            ratio = float(ratio)
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError('mi/me must be finite and positive')
            if self.masses['e'] is not None and self.masses['i'] is None:
                self.masses['i'] = float(self.masses['e']) * ratio
            elif self.masses['i'] is not None and self.masses['e'] is None:
                self.masses['e'] = float(self.masses['i']) / ratio
        self.charges = {'e': q_e, 'i': q_i}
        self.q_e, self.q_i = q_e, q_i
        self.m_e, self.m_i = self.masses['e'], self.masses['i']
        self.magnetic_scale = magnetic_scale
        self.velocity_source = velocity_source
        self.coordinate_offset = coordinate_offset
        self.squeeze = squeeze
        self.name = self.data_dirs['fields'].name
        # Only expose known grid/time metadata, never overwrite methods/paths.
        for key in ('nx', 'ny', 'nz', 'dx', 'dy', 'dz', 'lx', 'ly', 'lz', 'dt'):
            if key in self.params:
                setattr(self, key, self.params[key])
        self.steps = self._discover_steps()
        self.numslices = len(self.steps)
        self.format = 'hdf5'

    def _discover_steps(self):
        root = self.data_dirs['fields']
        steps = []
        for directory in root.glob('T.*'):
            suffix = directory.name[2:]
            if not directory.is_dir() or not suffix.isdigit():
                continue
            step = int(suffix)
            # Require the exact unpadded directory/file convention.
            if directory.name == f'T.{step}' and self._path('fields', step).is_file():
                steps.append(step)
        if not steps:
            raise FileNotFoundError(
                f'No snapshots found at {root}/T.<step>/fields_<step>.h5'
            )
        return tuple(sorted(steps))

    def _path(self, folder, step):
        if folder not in self.data_dirs:
            raise KeyError(f'No data_dirs path supplied for {folder!r}')
        return self.data_dirs[folder] / self.file_patterns[folder].format(step=step)

    def snapshot_path(self, folder, step):
        """Return the resolved filename for a folder key and simulation step."""
        return self._path(folder, step)

    def _group(self, h5, step):
        return h5[self.group_pattern.format(step=step)] if self.group_pattern else h5

    def available_variables(self, index=0):
        """Return actual dataset paths by folder for one output; no arrays read."""
        step = self._step_at(index)
        result = {}
        for folder in self.data_dirs:
            path = self._path(folder, step)
            if not path.is_file():
                result[folder] = []
                continue
            with h5py.File(path, 'r') as f:
                names = []
                self._group(f, step).visititems(
                    lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
                result[folder] = names
        return result

    def _step_at(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.numslices:
            raise IndexError(f'Snapshot index must be between 0 and {self.numslices - 1}; use load_step for timestep numbers')
        return self.steps[index]

    def load_xarray(self, slices, variables=('B', 'E')):
        """Load one snapshot index or iterable of indices (P3D-compatible API)."""
        if isinstance(slices, (int, np.integer)):
            return self.load_step(self._step_at(slices), variables)
        datasets = [self.load_step(self._step_at(index), variables) for index in slices]
        if not datasets:
            raise ValueError('slices must contain at least one snapshot')
        dim = 'time' if 'time' in datasets[0].attrs else 'step'
        expanded = [ds.expand_dims({dim: [ds.attrs[dim]]}) for ds in datasets]
        out = xr.concat(expanded, dim=dim, join='exact', combine_attrs='drop_conflicts')
        out = out.assign_coords(step=(dim, [ds.attrs['step'] for ds in datasets]))
        out.attrs.pop('snapshot', None)
        out.attrs.pop('step', None)
        out.attrs.pop('time', None)
        out.attrs.pop('time_simulation', None)
        if dim == 'time':
            out = out.assign_coords(time_simulation=(
                dim, [ds.attrs['time_simulation'] for ds in datasets]
            ))
            out['time'].attrs['units'] = datasets[0].attrs['time_units']
            out['time'].attrs['normalization'] = datasets[0].attrs['time_normalization']
            out['time_simulation'].attrs['units'] = 'simulation'
        return out

    snapshot = load_xarray

    def load_step(self, step, variables=('B', 'E')):
        """Load an actual simulation step, e.g. 40000, rather than its index."""
        if not isinstance(step, (int, np.integer)) or step not in self.steps:
            raise KeyError(f'Step {step!r} is unavailable; inspect run.steps')
        deps = dict(DEPENDENCIES)
        for species in ('e', 'i'):
            if self.masses[species] is None:
                deps.pop('P' + species)
        if self.velocity_source == 'direct':
            for species in ('e', 'i'):
                if 'P' + species in deps:
                    raise ValueError('Pressure reconstruction requires velocity_source=current')
        if self.velocity_source == 'direct':
            deps.update({f'u{s}': tuple(f'u{s}{d}' for d in AXES) for s in ('e', 'i')})
        if isinstance(variables, str):
            if variables in ('all', 'all_raw'):
                inventory = self.available_variables(self.steps.index(step))
                available = [name for name, (folder, dataset) in self.variable_map.items()
                             if dataset in inventory.get(folder, [])]
                if variables == 'all_raw':
                    variables = available
                else:
                    variables = [name for name, fields in deps.items()
                                 if set(fields) <= set(available)]
                    variables += [n for n in ('rhoe', 'rhoi', 'kee', 'kei') if n in available]
            else:
                variables = [variables]
        requested = list(variables)
        if not requested:
            raise ValueError('No variables requested or available through variable_map')
        for name in requested:
            if name in ('Pe', 'Pi') and name not in deps:
                raise ValueError(
                    f"{name} requires m_{name[-1]} in simulation units, supplied "
                    f"explicitly or in the info file; m_i can also come from mi/me; "
                    f"use T{name[-1]} to load raw stress"
                )
        native = list(dict.fromkeys(n for v in requested for n in deps.get(v, (v,))))
        missing = [n for n in native if n not in self.variable_map]
        if missing:
            raise ValueError(f'Add variable_map entries for: {missing}')
        raw, observed_shape = {}, None
        # Open each requested file once, and close before returning.
        routes = {}
        for name in native:
            folder, dataset = self.variable_map[name]
            routes.setdefault(folder, []).append((name, dataset))
        for folder, entries in routes.items():
            path = self._path(folder, step)
            with h5py.File(path, 'r') as f:
                group = self._group(f, step)
                for name, dataset in entries:
                    if dataset not in group:
                        raise KeyError(f'{dataset!r} (mapped from {name!r}) absent in {path}; inspect available_variables()')
                    array = group[dataset][...]
                    if array.ndim != len(self.axis_order):
                        raise ValueError(f'{name}: shape {array.shape} incompatible with axis_order={self.axis_order}; include singleton axes in axis_order')
                    if observed_shape is None:
                        observed_shape = array.shape
                    elif array.shape != observed_shape:
                        raise ValueError(f'{name}: shape {array.shape} differs from {observed_shape}; collocate fields/moments before analysis')
                    coords = self._coords(array.shape)
                    da = xr.DataArray(array, dims=self.axis_order, coords=coords)
                    da = da.transpose(*[d for d in AXES if d in da.dims])
                    if self.squeeze:
                        da = da.squeeze(drop=True)
                    if name in ('bx', 'by', 'bz'):
                        da = da / self.magnetic_scale
                    raw[name] = da
        raw = xr.Dataset(raw)
        out = xr.Dataset()
        for name in requested:
            if name in raw:
                out[name] = raw[name]
            elif name in ('ne', 'ni'):
                species = name[-1]
                out[name] = raw['rho' + species] / self.charges[species]
                out[name].attrs['definition'] = 'Number density: species charge density / charge'
            elif name == 'rho':
                out[name] = raw['rhoe'] + raw['rhoi']
                out[name].attrs['definition'] = 'Total charge density'
            elif name == 'J':
                out[name] = _vector(raw, deps['Je'], 'Je') + _vector(raw, deps['Ji'], 'Ji')
            elif name in ('Te', 'Ti'):
                out[name] = _tensor(raw, deps[name], name)
                out[name].attrs['definition'] = 'Uncentered VPIC stress tensor'
            elif name in ('Pe', 'Pi'):
                species = name[-1]
                mass = float(self.masses[species])
                if not np.isfinite(mass) or mass <= 0:
                    raise ValueError(f'm_{species} must be finite and positive')
                stress = _tensor(raw, deps['T' + species], name)
                velocity = self._velocity(raw, species)
                density = raw['rho' + species] / self.charges[species]
                bulk = velocity.rename(c='i') * velocity.rename(c='j')
                out[name] = (stress - mass * density * bulk).transpose(*stress.dims)
                out[name].attrs['definition'] = 'Nonrelativistic pressure: T - m*n*u*u'
                out[name].attrs['species_mass'] = mass
            elif name in ('ue', 'ui') and self.velocity_source == 'current':
                out[name] = self._velocity(raw, name[-1])
            else:
                out[name] = _vector(raw, deps[name][:3], name)
        out.attrs.update(run=self.name, snapshot=self.steps.index(step), step=int(step),
                         format='hdf5', axis_order=','.join(self.axis_order),
                         info_file=str(self.info_file), magnetic_scale=self.magnetic_scale,
                         q_e=self.q_e, q_i=self.q_i)
        if 'dt' in self.params:
            dt = float(self.params['dt'])
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError('dt must be finite and positive')
            simulation_time = float(self.params.get('t0', 0)) + step * dt
            out.attrs['time_simulation'] = simulation_time
            if 'wci' in self.params:
                wci = float(self.params['wci'])
                if not np.isfinite(wci) or wci <= 0:
                    raise ValueError('wci must be finite and positive')
                out.attrs['time'] = simulation_time * wci
                out.attrs['time_units'] = 'omega_ci^-1'
                out.attrs['time_normalization'] = 'omega_ci * simulation_time'
            else:
                out.attrs['time'] = simulation_time
                out.attrs['time_units'] = 'simulation'
                out.attrs['time_normalization'] = 'none (wci absent)'
        return out

    def _velocity(self, raw, species):
        current = _vector(raw, DEPENDENCIES['J' + species], 'u' + species)
        charge_density = raw['rho' + species]
        valid = charge_density / self.charges[species] > 0
        return current / charge_density.where(valid)

    def _coords(self, shape):
        coords = {}
        for axis, size in zip(self.axis_order, shape):
            declared = self.params.get('n' + axis)
            if declared is not None and float(declared) != size:
                raise ValueError(f'n{axis}={declared} disagrees with stored {axis} size {size}; check axis_order and global vs local grid')
            spacing = self.params.get('d' + axis)
            if spacing is None and 'l' + axis in self.params:
                spacing = float(self.params['l' + axis]) / size
            if spacing is None:
                values = np.arange(size)
                attrs = {'units': 'cell_index', 'coordinate_kind': 'index'}
            else:
                spacing = float(spacing)
                if not np.isfinite(spacing) or spacing <= 0:
                    raise ValueError(f'd{axis} must be finite and positive')
                origin = float(self.params.get(axis + '0', 0.0))
                values = origin + (np.arange(size) + self.coordinate_offset) * spacing
                attrs = {'coordinate_kind': 'physical', 'spacing': spacing}
            coords[axis] = xr.DataArray(values, dims=axis, attrs=attrs)
        return coords


def _vector(raw, names, name):
    first = raw[names[0]]
    return xr.concat([raw[n] for n in names], dim=xr.IndexVariable('c', list(AXES))).transpose(*first.dims, 'c').rename(name)


def _tensor(raw, names, name):
    xx, xy, xz, yy, yz, zz = [raw[n] for n in names]
    rows = [xr.concat(row, dim=xr.IndexVariable('j', list(AXES)))
            for row in ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz))]
    return xr.concat(rows, dim=xr.IndexVariable('i', list(AXES))).transpose(*xx.dims, 'i', 'j').rename(name)


# Optional lowercase alias for compatibility with older reader imports.
vpic = VPIC
