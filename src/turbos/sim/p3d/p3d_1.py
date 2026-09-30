#!/usr/bin/env python
###########################################################################
##
## The class to handle P3D simulation data in python. An object is created 
## that has the run parameters attached with it. The basic usage of the 
##  class can be done in the following ways.
##
## Example 1:
## > from p3d import p3d
## > r1=p3d('OTV')
## > r1.vars2load(['bx','by','bz'])
## > for i in range(20):
## >     r1.loadslice(i)
## >     DO WHATEVER!!!
## > r1.fin()
##
##
## Example 2:
## > from p3d import p3d
## > r1=p3d('OTV')
## > bxf=open('bx','rb')
## > bx=r1.readslice(bxf,10)
## > DO WHATEVER!!!
## > r1.fin()
##
## Methods in the class:
## 1) self.__init__(): Initialize the run object and load run parameters
## 2) self.print_params(): Print the run parameters
## 3) self.vars2load(): Define what variables to load, create variables
##    and open files for reading data.
## 4) self.readslice(): Read a time slice from an open file.
## 5) self.loadslice(): Load a snapshot of time for all the defined variables.
## 6) 
##
##                                        Tulasi Nandan Parashar
##                                        2014/08/23
## Hat tip to ColbyCH for teaching me the basics of classes in python.
##
###########################################################################

import numpy as np
from os.path import basename, realpath, exists
from pathlib import Path
from scipy.ndimage import gaussian_filter as gf


def calc_dep(requested, primitives, derived):
   """Resolve primitive and derived fields in dependency order."""
   result = []
   visited = set()
   active = set()
   # These names describe vector collections; their components are loaded
   # individually, so no placeholder array is added for the collection.
   groups = {'vi', 've', 'omi', 'ome', 'om', 'zpzm'}

   def visit(name):
      if name in active:
         raise ValueError(f'Cyclic derived-variable dependency at {name!r}')
      if name in visited:
         return
      if name not in primitives and name not in derived:
         raise ValueError(f'Unknown P3D variable {name!r}')
      active.add(name)
      for dependency in derived.get(name, ()):
         visit(dependency)
      active.remove(name)
      visited.add(name)
      if name not in groups:
         result.append(name)

   for name in requested:
      visit(name)
   return result


def _periodic_derivative(array, spacing, axis):
   """Centered first derivative; an invariant axis has zero derivative."""
   if array.shape[axis] == 1:
      return np.zeros_like(array)
   return (np.roll(array, -1, axis=axis) - np.roll(array, 1, axis=axis)) / (2 * spacing)


def _curl(ax, ay, az, dx, dy, dz):
   return (
      _periodic_derivative(az, dy, 1) - _periodic_derivative(ay, dz, 2),
      _periodic_derivative(ax, dz, 2) - _periodic_derivative(az, dx, 0),
      _periodic_derivative(ay, dx, 0) - _periodic_derivative(ax, dy, 1),
   )


def _curl_component(component, vx, vy, vz, dx, dy, dz):
   """A curl component needs only its two transverse input fields."""
   if component == 'x':
      return _periodic_derivative(vz, dy, 1) - _periodic_derivative(vy, dz, 2)
   if component == 'y':
      return _periodic_derivative(vx, dz, 2) - _periodic_derivative(vz, dx, 0)
   if component == 'z':
      return _periodic_derivative(vy, dx, 0) - _periodic_derivative(vx, dy, 1)
   raise ValueError(f'Invalid curl component {component!r}')


def _divergence(ax, ay, az, dx, dy, dz):
   return (_periodic_derivative(ax, dx, 0) +
           _periodic_derivative(ay, dy, 1) +
           _periodic_derivative(az, dz, 2))


class p3d(object):
   """p3d object:
         Tulasi Parashar's version to read data from
         P3D github version.
         Created on 08/22/2014
         Last modified on 09/09/2015
         Last modified on 10/13/2016
   """
   def __init__(self,shelldirname=None,filenum=None):
      # If no rundir specified
      if shelldirname is None: 
         shelldirname = input('Please enter the rundir: ') 
      self.rundir = realpath(shelldirname)
      self.dirname= basename(self.rundir)

      # If filenum not given
      if filenum is None:
         self.filenum=input("Please enter the file number to load (e.g. 000): ")
      else:
         self.filenum=filenum

      # Check where the paramfile is
      if exists(self.rundir+'/param_'+self.dirname):
         self.paramfile=self.rundir+'/param_'+self.dirname
      elif exists(self.rundir+'/staging/param_'+self.dirname):
         self.paramfile=self.rundir+'/staging/param_'+self.dirname
      elif exists(self.rundir+'/paramfile'):
         self.paramfile=self.rundir+'/paramfile'
      else:
         raise ValueError('Paramfile not found in '+self.dirname)
      # load parameters
      self.params=loadparams(self.paramfile)
      for i in list(self.params.keys()):
         self.__dict__[i]=self.params[i]

      self.primitives=['bx','by','bz','ex','ey','ez','jix','jiy','jiz','jex','jey','jez','jx','jy','jz','ni','pixx','piyy','pizz','pixy','pixz','piyz','pexx','pexy','pexz','peyy','peyz','pezz','ne','rho']
      self.field=['bx','by','bz']
      # self.primitives=['jix','jiy','jiz','jex','jey','jez','pixx','piyy','pizz','pixy','pixz','piyz','pexx','pexy','pexz','peyy','peyz','pezz', 'ne','ni']
      self.derived={\
      'tex':['pexx','ne'],'tey':['peyy','ne'],'tez':['pezz','ne'],\
      'te':['tex','tey','tez'],\
      'tix':['pixx','ni'],'tiy':['piyy','ni'],'tiz':['pizz','ni'],\
      'ti':['tix','tiy','tiz'],\
      'vix':['jix','ni'],'viy':['jiy','ni'],'viz':['jiz','ni'],\
      'vi':['vix','viy','viz'],\
      'vex':['jex','ne'],'vey':['jey','ne'],'vez':['jez','ne'],\
      've':['vex','vey','vez'],\
      'omix':['viy','viz'],'omiy':['viz','vix'],'omiz':['vix','viy'],\
      'omi':['omix','omiy','omiz'],'ensti':['omi'],'pali':['omi'],\
      'omex':['vey','vez'],'omey':['vex','vez'],'omez':['vex','vey'],\
      'ome':['omex','omey','omez'],'enste':['ome'],'pale':['ome'],\
      'omx':['cmy','cmz'],'omy':['cmz','cmx'],'omz':['cmx','cmy'],\
      'om':['omx','omy','omz'], 'enst':['om'], 'pal':['om'],\
      'dui':['vix','viy','viz'],'due':['vex','vey','vez'],\
      'den':['ni','ne'],\
      'cmx':['vix','vex'],'cmy':['viy','vey'],'cmz':['viz','vez'],\
      'zpx':['bx','cmx','den'],'zpy':['by','cmy','den'],'zpz':['bz','cmz','den'],\
      'zmx':['bx','cmx','den'],'zmy':['by','cmy','den'],'zmz':['bz','cmz','den'],\
      'zpzm':['zpx','zpy','zpz','zmx','zmy','zmz']\
      }

      self.allvars=self.primitives+list(self.derived.keys())
      snapshots = sorted(Path(self.rundir).glob('*.???.dat'))
      snapshots = [p for p in snapshots if p.stem.rsplit('.', 1)[-1].isdigit()]
      self.storage_format = 'movie'
      if snapshots:
         with snapshots[0].open('rb') as file:
            if file.read(8) == b'\x89HDF\r\n\x1a\n':
               prefixes = {p.name.rsplit('.', 2)[0] for p in snapshots}
               if self.dirname in prefixes:
                  self._hdf_prefix = self.dirname
               elif len(prefixes) == 1:
                  self._hdf_prefix = prefixes.pop()
               else:
                  raise ValueError(f'Multiple HDF5 snapshot prefixes found: {sorted(prefixes)}')
               self.storage_format = 'hdf5'


####
#### Method to print the parameters associated with the run
####
   def print_params(self):
      """
         A quick method to print the parameters and variables attached with
         the p3d run object.
      """
      for i in list(self.params.keys()):
         print(i,' = ',self.params[i])
####
#### Method to define the variables to load, create variables and
#### open corresponding files.
#### 
   def vars2load(self,v2lu):
      """
         Define the variables to load, define corresponding numpy arrays &
         open the files
      """
      if isinstance(v2lu, str):
         v2lu = [v2lu]
      shortcuts = {
         'min': ['bx','by','bz','jix','jiy','jz','ni'],
         'prim': self.primitives,
         'field': self.field,
         'all': self.allvars,
      }
      if not v2lu:
         raise ValueError('Choose at least one variable to load')
      requested = shortcuts.get(v2lu[0], v2lu) if len(v2lu) == 1 else v2lu
      self.vars2l = calc_dep(requested, self.primitives, self.derived)
      if hasattr(self, 'logfile') and not self.logfile.closed:
         self.logfile.close()
      for name in self.primitives:
         handle = self.__dict__.get(name+'f')
         if handle is not None and not handle.closed:
            handle.close()

      # If byte or double byte data, open the log file.
      if self.data_type in ("b", "bb"):
         if exists(self.rundir+'/log'):
            print(self.rundir+'/log')
            self.logfile=open(self.rundir+"/log","r")
         else:
            print(self.rundir+'/staging/movie.log.'+self.filenum)
            self.logfile=open(self.rundir+"/staging/movie.log."+self.filenum,"r")
         self.szl=np.size(self.logvars)
         self.alllogvals=np.loadtxt(self.logfile)

      # Create arrays and open files
      for i in self.vars2l:
         if i in self.primitives:
            if exists(self.rundir+'/'+i):
               self.__dict__[i+'f']=open(self.rundir+'/'+i,"rb")
            else:
               self.__dict__[i+'f']=open(self.rundir+'/staging/movie.'+i+\
                                    '.'+str(self.filenum).zfill(3),"rb")
      # If 'b' or 'bb' data type, load minmax for each loaded variable
      if self.data_type in ('b', 'bb'):
         for i in self.vars2l:
            if i in self.primitives:
               self.__dict__[i+'minmax']=self.alllogvals[self.logvars.index(i)\
                                         ::len(self.logvars),:]
      # Find out the number of slices for the open file
      self.__dict__[self.vars2l[0]+'f'].seek(0,2)
      filesize=self.__dict__[self.vars2l[0]+'f'].tell()
      numbersize=2**['b','bb','f','d'].index(self.data_type)
      self.numslices=filesize//(self.nx*self.ny*self.nz*numbersize)
####
#### Method to read a particular time slice from an open file.
#### 
   def readslice(self,f,timeslice,v=""):
      """
         This method reads a particular slice of time from a given file. The
         explicit inputs are file name, time slice and data type. It is used 
         as:
            output=self.readslice(filename,time)
      """
      ### Byte data #########################################################
      if self.data_type == 'b':
         f.seek(timeslice*self.nx*self.ny*self.nz)
         field = np.fromfile(f,dtype='uint8',count=self.nx*self.ny*self.nz)
         field = np.reshape(field,(self.nx,self.ny,self.nz),order='F')
         minmax=self.__dict__[v+'minmax'][timeslice]
         field = minmax[0]+(minmax[1]-minmax[0])*field/255.
      ### Double byte data #################################################
      elif self.data_type == 'bb':
         f.seek(2*timeslice*self.nx*self.ny*self.nz)
         field = np.fromfile(f,dtype='int16',count=self.nx*self.ny*self.nz)
         field = np.reshape(field,(self.nx,self.ny,self.nz),order='F')
         minmax=self.__dict__[v+'minmax'][timeslice]
         field = minmax[0]+(minmax[1]-minmax[0])*(field+32678.)/65535.
      ### Four byte (single precision) data ################################
      elif self.data_type == 'f':
         f.seek(4*timeslice*self.nx*self.ny*self.nz)
         field = np.fromfile(f,dtype='float32',count=self.nx*self.ny*self.nz)
         field = np.reshape(field,(self.nx,self.ny,self.nz),order='F').astype('float64')
      ### Double precision data ############################################
      elif self.data_type == 'd':
         f.seek(8*timeslice*self.nx*self.ny*self.nz)
         field = np.fromfile(f,dtype='float64',count=self.nx*self.ny*self.nz)
         field = np.reshape(field,(self.nx,self.ny,self.nz),order='F')
      return field
####
#### Method to load time slices for the loaded variables.
####
   def loadslice(self,it,smth=None):
      """
         Load the variables initialized by self.vars2load()
      """
      for i in self.vars2l:
         if i in self.primitives:
            self.__dict__[i]=self.readslice(self.__dict__[i+'f'],it,i)
      for i in self.vars2l:
         if i in self.derived:
           #self.__dict__[i] = self._derivedv(i)
            self._derivedv(i)

      self.mmd={}
      for i in self.vars2l:
         if smth is not None:
            self.__dict__[i]=gf(self.__dict__[i],sigma=smth)
         self.mmd[i]=[self.__dict__[i].min(),self.__dict__[i].max()]
      self.time = it*self.dtmovie

   def to_xarray(self, variables=None, squeeze=True):
      """Wrap the currently loaded snapshot in an xarray.Dataset.

      Call ``vars2load`` and ``loadslice`` first. Arrays retain the P3D
      dimension order (x, y, z); singleton axes can be removed for 2D runs.
      The Dataset shares the loaded NumPy arrays where possible.
      """
      import xarray as xr

      if not hasattr(self, 'time'):
         raise RuntimeError('Call loadslice() before to_xarray().')
      if variables is None:
         variables = [name for name in self.vars2l
                      if isinstance(getattr(self, name, None), np.ndarray)
                      and getattr(self, name).shape == (self.nx, self.ny, self.nz)]
      elif isinstance(variables, str):
         variables = [variables]

      dims = ('x', 'y', 'z')
      coords = getattr(self, '_snapshot_coords', None)
      if coords is None:
         coords = {dim: np.arange(getattr(self, 'n' + dim)) * getattr(self, 'd' + dim)
                   for dim in dims}
      shape = (self.nx, self.ny, self.nz)
      data = {}
      for name in variables:
         value = getattr(self, name)
         if not isinstance(value, np.ndarray) or value.shape != shape:
            raise ValueError(f'{name!r} is not a loaded field with shape {shape}')
         data[name] = (dims, value)
      ds = xr.Dataset(data, coords=coords,
                      attrs={'time': self.time, 'run': self.dirname,
                             'slice': getattr(self, '_snapshot_id', round(self.time / self.dtmovie)),
                             'dtmovie': self.dtmovie})
      return ds.squeeze(drop=True) if squeeze else ds

   def _load_hdf5_snapshot(self, index, variables, smth=None, squeeze=True):
      """Read one numbered HDF5 snapshot and wrap selected fields in xarray."""
      import h5py

      path = Path(self.rundir) / f'{self._hdf_prefix}.{index:03d}.dat'
      if not path.is_file():
         raise FileNotFoundError(f'HDF5 snapshot not found: {path}. '
                                 'Use the number in the filename, e.g. 9 for .009.dat.')
      if isinstance(variables, str):
         variables = [variables]
      if variables is None:
         variables = getattr(self, 'vars2l', None)
      if not variables:
         raise ValueError('Provide variables for the HDF5 snapshot')
      shortcuts = {'min': ['bx','by','bz','jix','jiy','jz','ni'],
                   'prim': self.primitives, 'field': self.field,
                   'all': self.allvars}
      requested = shortcuts.get(variables[0], variables) if len(variables) == 1 else variables
      self.vars2l = calc_dep(requested, self.primitives, self.derived)
      with h5py.File(path, 'r') as h5:
         self._snapshot_coords = {dim: np.asarray(h5[name][()])
                                  for dim, name in (('x','xx'),('y','yy'),('z','zz'))}
         shape = (self.nx, self.ny, self.nz)
         for dim, length in zip(('x','y','z'), shape):
            if len(self._snapshot_coords[dim]) != length:
               raise ValueError(f'{path}: {dim} coordinate disagrees with paramfile')
         for name in self.vars2l:
            if name in self.primitives:
               if name not in h5:
                  raise KeyError(f'{name!r} not found in {path}')
               if h5[name].shape != shape:
                  raise ValueError(f'{name!r} has shape {h5[name].shape}, expected {shape}')
               self.__dict__[name] = h5[name][...]
         self.time = float(h5['time'][()])
      for name in self.vars2l:
         if name in self.derived:
            self._derivedv(name)
      for name in self.vars2l:
         if smth is not None:
            self.__dict__[name] = gf(self.__dict__[name], sigma=smth)
      self._snapshot_id = int(index)
      return self.to_xarray(squeeze=squeeze)

   def load_xarray(self, slices, variables=None, smth=None, squeeze=True):
      """Load one or several snapshots as an xarray.Dataset.

      ``variables`` uses the same names/shortcuts as ``vars2load``.
      A single integer returns spatial fields; multiple indices add a time
      dimension. Load large series in smaller batches to limit memory use.
      """
      import xarray as xr

      if self.storage_format == 'hdf5':
         if isinstance(slices, (int, np.integer)):
            return self._load_hdf5_snapshot(int(slices), variables, smth, squeeze)
         indices = list(slices)
         if not indices:
            raise ValueError('slices must contain at least one snapshot number')
         return xr.concat([
            self._load_hdf5_snapshot(int(index), variables, smth, squeeze)
            .expand_dims(time=[self.time]) for index in indices], dim='time')

      if variables is not None:
         if isinstance(variables, str):
            variables = [variables]
         self.vars2load(variables)
      elif not hasattr(self, 'vars2l'):
         raise ValueError('Provide variables or call vars2load() first.')

      if isinstance(slices, (int, np.integer)):
         self.loadslice(int(slices), smth=smth)
         return self.to_xarray(squeeze=squeeze)

      indices = list(slices)
      if not indices:
         raise ValueError('slices must contain at least one index')
      snapshots = []
      for index in indices:
         self.loadslice(int(index), smth=smth)
         snapshots.append(self.to_xarray(squeeze=squeeze).expand_dims(time=[self.time]))
      return xr.concat(snapshots, dim='time')

####
#### Method to add attributes to the object
####
   def addattr(self,key,val):
      for i in key:
         print('Adding '+i) 
         self.__dict__[i]=val[key.index(i)]
         if isinstance(val[key.index(i)],np.ndarray):
            self.mmd[i]=[self.__dict__[i].min(),self.__dict__[i].max()]

####
#### Method to compute derived quantities
####
   def _derivedv(self,varname):
      if varname == 'tix'   : self.tix    = self.pixx/self.ni
      if varname == 'tiy'   : self.tiy    = self.piyy/self.ni
      if varname == 'tiz'   : self.tiz    = self.pizz/self.ni
      if varname == 'ti'    : self.ti     = (self.tix+self.tiy+self.tiz)/3.
      if varname == 'tex'   : self.tex    = self.pexx/self.ne
      if varname == 'tey'   : self.tey    = self.peyy/self.ne
      if varname == 'tez'   : self.tez    = self.pezz/self.ne
      if varname == 'te'    : self.te     = (self.tex+self.tey+self.tez)/3.
      if varname == 'vix'   : self.vix    = self.jix/self.ni
      if varname == 'viy'   : self.viy    = self.jiy/self.ni
      if varname == 'viz'   : self.viz    = self.jiz/self.ni
      if varname in ('omix', 'omiy', 'omiz'):
         self.__dict__[varname] = _curl_component(varname[-1],
            getattr(self,'vix',None),getattr(self,'viy',None),getattr(self,'viz',None),self.dx,self.dy,self.dz)
      if varname == 'omi'   : pass
      if varname == 'ensti' : self.ensti  = self.omix**2+self.omiy**2+self.omiz**2
      if varname == 'dui'   : self.dui    = _divergence(self.vix,self.viy,self.viz,self.dx,self.dy,self.dz)
      if varname == 'vex'   : self.vex    = -self.jex/self.ne
      if varname == 'vey'   : self.vey    = -self.jey/self.ne
      if varname == 'vez'   : self.vez    = -self.jez/self.ne
      if varname in ('omex', 'omey', 'omez'):
         self.__dict__[varname] = _curl_component(varname[-1],
            getattr(self,'vex',None),getattr(self,'vey',None),getattr(self,'vez',None),self.dx,self.dy,self.dz)
      if varname == 'ome'   : pass
      if varname == 'enste' : self.enste  = self.omex**2+self.omey**2+self.omez**2
      if varname == 'due'   : self.due    = _divergence(self.vex,self.vey,self.vez,self.dx,self.dy,self.dz)
      if varname == 'cmx'   : self.cmx    = (self.vix+self.m_e*self.vex)/(1+self.m_e)
      if varname == 'cmy'   : self.cmy    = (self.viy+self.m_e*self.vey)/(1+self.m_e)
      if varname == 'cmz'   : self.cmz    = (self.viz+self.m_e*self.vez)/(1+self.m_e)
      if varname in ('omx', 'omy', 'omz'):
         self.__dict__[varname] = _curl_component(varname[-1],
            getattr(self,'cmx',None),getattr(self,'cmy',None),getattr(self,'cmz',None),self.dx,self.dy,self.dz)
      if varname == 'om'    : pass
      if varname == 'enst'  : self.enst   = self.omx**2+self.omy**2+self.omz**2
      if varname == 'den'   : self.den    = self.ni+self.m_e*self.ne
      if varname == 'zpx'   : self.zpx    = self.bx/np.sqrt(self.den) + self.cmx
      if varname == 'zpy'   : self.zpy    = self.by/np.sqrt(self.den) + self.cmy
      if varname == 'zpz'   : self.zpz    = self.bz/np.sqrt(self.den) + self.cmz
      if varname == 'zmx'   : self.zmx    = self.bx/np.sqrt(self.den) - self.cmx
      if varname == 'zmy'   : self.zmy    = self.by/np.sqrt(self.den) - self.cmy
      if varname == 'zmz'   : self.zmz    = self.bz/np.sqrt(self.den) - self.cmz
      if varname == 'zpzm'  : pass

      if varname == 'pali'  : 
         tmp = _curl(self.omix,self.omiy,self.omiz,self.dx,self.dy,self.dz)
         self.pali   = 0.5*(tmp[0]**2+tmp[1]**2+tmp[2]**2); tmp=None
      if varname == 'pale'  : 
         tmp = _curl(self.omex,self.omey,self.omez,self.dx,self.dy,self.dz)
         self.pale   = 0.5*(tmp[0]**2+tmp[1]**2+tmp[2]**2); tmp=None
      if varname == 'pal'  : 
         tmp = _curl(self.omx,self.omy,self.omz,self.dx,self.dy,self.dz)
         self.pal    = 0.5*(tmp[0]**2+tmp[1]**2+tmp[2]**2); tmp=None
####
#### Method to close opened files.
####
   def fin(self):
      """
         close the run files.
      """
      for i in getattr(self, 'vars2l', ()):
         if i in self.primitives:
            handle = self.__dict__.get(i+'f')
            if handle is not None:
               handle.close()
      if hasattr(self, 'logfile'):
         self.logfile.close()
####
#### Load energies for the run
####
   def loadenergies(self):
      """
         Loads the energies for the run object along with four different
         time series: 
         self.t -> Time in cyclotron units 
         self.tnl -> Time in units of Nominal nonlinear time 
                     based on initial energy
         self.ltnl -> Local Nonlinear time throughout the simulation
         self.ta -> Turbulence age based on the local nonlinear time.
      """
      self.evars=['t', 'eges', 'ebx' , 'eby' , 'ebz' , 'eex' , 'eey' , 'eez' ,\
      'eem' , 'ekix', 'ekiy', 'ekiz', 'ekex', 'ekey', 'ekez', 'ekin', 'eifx', \
      'eify', 'eifz', 'eefx', 'eefy', 'eefz', 'eipx', 'eipy', 'eipz', 'eepx', \
      'eepy', 'eepz']
      data=np.loadtxt(self.rundir+'/Energies.dat')
      for i in self.evars:
         self.__dict__[i]=data[:,self.evars.index(i)]
      self.eb0=0.5*(self.b0x**2+self.b0y**2+self.b0z**2)
      self.eb =self.ebx +self.eby +self.ebz
      self.eip=self.eipx+self.eipy+self.eipz
      self.eep=self.eepx+self.eepy+self.eepz
      self.eif=self.eifx+self.eify+self.eifz
      self.eef=self.eefx+self.eefy+self.eefz
      self.ee =self.eex +self.eey +self.eez
      self.edz=self.eb-self.eb0+self.eif+self.eef
     #self.tnl=self.t*np.sqrt(2*self.edz[0])*2*np.pi/self.lx
      self.tnl=self.t*np.sqrt(4*self.edz[0])*2*np.pi/self.lx
      self.ltnl=self.lx/(np.sqrt(self.edz)*4*np.pi)
      self.ta=np.zeros(len(self.eb))
      for i in range(1,len(self.eb)):
         self.ta[i]=self.ta[i-1]+self.dt*2./(self.ltnl[i-1]+self.ltnl[i])

###
### Method to load parameters
###
def loadparams(paramfile):
   params={}
   def _convert(val):
       constructors = [int, float, str]
       for c in constructors:
           try:
               return c(val)
           except ValueError:
               pass

   with open(paramfile) as f: 
      content = f.readlines()

   for item in content:
      if '#define' in item and item[0] != '!':
         if len(item.split()) > 2:
            key = item.split()[1]
            val = item.split()[2]
            val = _convert(item.split()[2])
         else:
            key = item.split()[1]
            val = True
         params[key] = val

   ## For hybrid code, set electron mass to extremely small 
   ## and speed of light to extremely large.
   if 'hybrid' in params: 
      params['c_2'] = 1e9
      if 'd_e2' in params:
         params['m_e'] = params['d_e2']
      else:
         params['m_e'] = 0.000545
   for i in ['b0x','b0y','b0z']:
      if i not in params:
         params[i]=0.
   
   # Set data type
   if 'eight_byte' in params: params['data_type']='d'
   elif 'four_byte' in params: params['data_type']='f'
   elif 'double_byte' in params: params['data_type']='bb'
   else: params['data_type']='b'
   #
   #
   if params['movie_header'] == '"movie2dC.h"':
      params['logvars']=['rho', 'jx','jy','jz', 'bx','by','bz', 'ex','ey','ez',
      'ne', 'jex','jey','jez', 'pexx','peyy','pezz','pexy','peyz','pexz',
      'ni', 'pixx','piyy','pizz','pixy','piyz','pixz']
   elif params['movie_header'] == '"movie4b.h"':
      params['logvars']=['rho', 'jx','jy','jz', 'bx','by','bz', 'ex','ey','ez',
      'ne','jex','jey','jez', 'pexx','peyy','pezz','pexz','peyz','pexy',
      'ni','jix','jiy','jiz', 'pixx','piyy','pizz','pixz','piyz','pixy']
   elif params['movie_header'] == '"movie2dD.h"':
      params['logvars']=['rho', 'jx','jy','jz', 'bx','by','bz', 'ex','ey','ez',
      'ne', 'jex','jey','jez', 'pexx','peyy','pezz','pexy','peyz','pexz',
      'ni', 'jix','jiy','jiz', 'pixx','piyy','pizz','pixy','piyz','pixz']
   elif params['movie_header'] == '"movie3dHeat.h"':
      params['logvars']=['rho', 'jx','jy','jz', 'bx','by','bz', 'ex','ey','ez',
      'ne', 'jex','jey','jez', 'pexx','peyy','pezz','pexy', 'peyz', 'pexz',
      'ni', 'pixx','piyy','pizz','pixy', 'piyz', 'pixz', 'epar1','epar2',
      'epar3','eperp1','eperp2','eperp3', 'vpar1','vpar2','vpar3']
   elif params['movie_header'] == '"movie_pic3.0.h"':
      params['logvars']=['rho', 'jx','jy','jz', 'bx','by','bz', 'ex','ey','ez',
      'ne', 'jex','jey','jez', 'pexx','peyy','pezz','pexy','peyz','pexz',
      'ni', 'jix','jiy','jiz', 'pixx','piyy','pizz','pixy','piyz','pixz']
   else:
      print('='*80 + \
                '\t This particular moive headder has not been coded!\n'\
                '\t Talk to Tulasi to fit it, or fix it yourself.\n'\
                '\t I dont care, Im a computer not a cop'\
                '='*80)
   #
   #
   # Derive some others
   if 'n_movieout' in params:
      params['dtmovie']=params['n_movieout']*params['dt']
   else:
      params['dtmovie']=params['movieout']
   params['nx']=int(params['pex']*params['nx'])
   params['ny']=int(params['pey']*params['ny'])
   params['nz']=int(params['pez']*params['nz'])
   params['dx']=params['lx']/params['nx']
   params['dy']=params['ly']/params['ny']
   params['dz']=params['lz']/params['nz']
   params['xx']=np.linspace(0.,params['lx'],params['nx'])
   params['yy']=np.linspace(0.,params['ly'],params['ny'])
   params['zz']=np.linspace(0.,params['lz'],params['nz'])
   params['xxt']=np.linspace(0.,2*np.pi,params['nx'])
   params['yyt']=np.linspace(0.,2*np.pi,params['ny'])
   params['zzt']=np.linspace(0.,2*np.pi,params['nz'])
   if all([i in params for i in ['b0x','b0y','b0z']]):
      params['b0'] =np.sqrt(params['b0x']**2+params['b0y']**2+params['b0z']**2)
#   params['betai'] = 2*params['n_0']*params['T_i']/params['b0']**2
#  params['betae'] = 2*params['n_0']*params['T_e']/params['b0']**2
   params['nprocs'] = int(params['pex']*params['pey']*params['pez'])
#   params['lambdae']=np.sqrt(params['T_e']/(params['n_0']*params['c_2']))
   if params['m_e'] != 0:
      params['wce']=params['b0']/params['m_e']
      params['vthe']=np.sqrt(2*params['T_e']/params['m_e'])
   else:
      params['wce']=1e9
      params['vthe']=1e9
   params['de']=np.sqrt(params['m_e'])
   params['rhoe']=params['vthe']/params['wce']
   params['wpe']=np.sqrt(params['c_2'])/params['de']
   params['wpi']=np.sqrt(params['c_2'])
   params['vthi']=np.sqrt(2*params['T_i'])
   params['wci']=params['b0']
   params['rhoi']=params['vthi']/params['wci']
   params['kgrid'] = max(np.pi/params['dx'], np.pi/params['dy'], np.pi/params['dz'])
#   params['ca']=params['b0']/np.sqrt(params['n_0']) 
   
   return params
