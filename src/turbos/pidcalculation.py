import numpy as np
#from TurbAn.Analysis.Simulations import AnalysisFunctions as af
import Turbcopy.AnalysisFunctions as af
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import uniform_filter

def calcD(rc,sp, dx):
	# if avg == 'gf':
	# 	if sp == 'e':
	# 		uex = gf((-rc.jex/rc.ne)[..., 0], sigma=window, mode="wrap")
	# 		uey = gf((-rc.jey/rc.ne)[..., 0], sigma=window, mode="wrap")
	# 		uez = gf((-rc.jez/rc.ne)[..., 0], sigma=window, mode="wrap")
	# 	if sp == 'i':
	# 		uix = gf((-rc.jix/rc.ni)[..., 0], sigma=window, mode="wrap")
	# 		uiy = gf((-rc.jiy/rc.ni)[..., 0], sigma=window, mode="wrap")
	# 		uiz = gf((-rc.jiz/rc.ni)[..., 0], sigma=window, mode="wrap")

	# if avg == 'box':
	# 	for k in range(3):
	# 		if sp == 'e':
	# 			uex = boxcar((-rc.jex/rc.ne)[..., 0], window =window)
	# 			uey = boxcar((-rc.jey/rc.ne)[..., 0], window =window)
	# 			uez = boxcar((-rc.jez/rc.ne)[..., 0], window =window)
	# 		if sp == 'i':
	# 			uix = boxcar((-rc.jix/rc.ni)[..., 0], window =window)
	# 			uiy = boxcar((-rc.jiy/rc.ni)[..., 0], window =window)
	# 			uiz = boxcar((-rc.jiz/rc.ni)[..., 0], window =window)
	
	# if avg == None:
	if sp == 'e':
		uex = -(rc.jex/rc.ne)
		uey = -(rc.jey/rc.ne)
		uez = -(rc.jez/rc.ne)
	if sp == 'i':
		uix = (rc.jix/rc.ni)
		uiy = (rc.jiy/rc.ni)
		uiz = (rc.jiz/rc.ni)
			

	if sp == 'e':
		dxux=af.pderiv(rc.uex,dx=dx,ax=0,order=2,smth=None)
		dxuy=af.pderiv(rc.uey,dx=dx,ax=0,order=2,smth=None)
		dxuz=af.pderiv(rc.uez,dx=dx,ax=0,order=2,smth=None)
		dyux=af.pderiv(rc.uex,dx=dx,ax=1,order=2,smth=None)
		dyuy=af.pderiv(rc.uey,dx=dx,ax=1,order=2,smth=None)
		dyuz=af.pderiv(rc.uez,dx=dx,ax=1,order=2,smth=None)
	if sp == 'i':
		dxux=af.pderiv(rc.uix,dx=dx,ax=0,order=2,smth=None)
		dxuy=af.pderiv(rc.uiy,dx=dx,ax=0,order=2,smth=None)
		dxuz=af.pderiv(rc.uiz,dx=dx,ax=0,order=2,smth=None)
		dyux=af.pderiv(rc.uix,dx=dx,ax=1,order=2,smth=None)
		dyuy=af.pderiv(rc.uiy,dx=dx,ax=1,order=2,smth=None)
		dyuz=af.pderiv(rc.uiz,dx=dx,ax=1,order=2,smth=None)		

	Sxx=dxux
	Sxy=0.5*(dxuy+dyux)
	Syx=0.5*(dxuy+dyux)
	Syy=dyuy
	Sxz=0.5*(dxuz)
	Szx=0.5*(dxuz)
	Syz=0.5*(dyuz)
	Szy=0.5*(dyuz)
	Szz=0.

	theta=Sxx+Syy
	
	Dxx=Sxx-(theta)/3.
	Dxy=Sxy
	Dxz=Sxz

	Dyy=Syy-(theta)/3.
	Dyx=Syx
	Dyz=Syz
	
	Dzz=Szz-(theta)/3.
	Dzx=Szx
	Dzy=Szy

	return Dxx,Dxy,Dxz,Dyy,Dyx,Dyz,Dzz,Dzx,Dzy, theta


def calcPI(rc,sp):
	# if avg == 'gf':
	# 	if sp == 'e':
	# 		Pxx=gf(rc.pexx[..., 0], sigma = window, mode = 'wrap')
	# 		Pxy=gf(rc.pexy[..., 0], sigma = window, mode = 'wrap')
	# 		Pxz=gf(rc.pexz[..., 0], sigma = window, mode = 'wrap')
	# 		Pyy=gf(rc.peyy[..., 0], sigma = window, mode = 'wrap')
	# 		Pyx=gf(rc.pexy[..., 0], sigma = window, mode = 'wrap')
	# 		Pyz=gf(rc.peyz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzz=gf(rc.pezz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzx=gf(rc.pexz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzy=gf(rc.peyz[..., 0], sigma = window, mode = 'wrap')
	# 	if sp == 'i':
	# 		Pxx=gf(rc.pixx[..., 0], sigma = window, mode = 'wrap')
	# 		Pxy=gf(rc.pixy[..., 0], sigma = window, mode = 'wrap')
	# 		Pxz=gf(rc.pixz[..., 0], sigma = window, mode = 'wrap')
	# 		Pyy=gf(rc.piyy[..., 0], sigma = window, mode = 'wrap')
	# 		Pyx=gf(rc.pixy[..., 0], sigma = window, mode = 'wrap')
	# 		Pyz=gf(rc.piyz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzz=gf(rc.pizz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzx=gf(rc.pixz[..., 0], sigma = window, mode = 'wrap')
	# 		Pzy=gf(rc.piyz[..., 0], sigma = window, mode = 'wrap')

	# if avg == 'box':
	# 	for k in range(3):
	# 		if sp == 'e':
	# 			Pxx=boxcar(rc.pexx[..., 0], window = window)
	# 			Pxy=boxcar(rc.pexy[..., 0], window = window)
	# 			Pxz=boxcar(rc.pexz[..., 0], window = window)
	# 			Pyy=boxcar(rc.peyy[..., 0], window = window)
	# 			Pyx=boxcar(rc.pexy[..., 0], window = window)
	# 			Pyz=boxcar(rc.peyz[..., 0], window = window)
	# 			Pzz=boxcar(rc.pezz[..., 0], window = window)
	# 			Pzx=boxcar(rc.pexz[..., 0], window = window)
	# 			Pzy=boxcar(rc.peyz[..., 0], window = window)
	# 		if sp == 'i':
	# 			Pxx=boxcar(rc.pixx[..., 0], window = window)
	# 			Pxy=boxcar(rc.pixy[..., 0], window = window)
	# 			Pxz=boxcar(rc.pixz[..., 0], window = window)
	# 			Pyy=boxcar(rc.piyy[..., 0], window = window)
	# 			Pyx=boxcar(rc.pixy[..., 0], window = window)
	# 			Pyz=boxcar(rc.piyz[..., 0], window = window)
	# 			Pzz=boxcar(rc.pizz[..., 0], window = window)
	# 			Pzx=boxcar(rc.pixz[..., 0], window = window)
	# 			Pzy=boxcar(rc.piyz[..., 0], window = window)

	# if avg == None:
	if sp == 'e':
		Pxx=rc.pexx
		Pxy=rc.pexy
		Pxz=rc.pexz
		Pyy=rc.peyy
		Pyx=rc.pexy
		Pyz=rc.peyz
		Pzz=rc.pezz
		Pzx=rc.pexz
		Pzy=rc.peyz
	if sp == 'i':
		Pxx=rc.pixx
		Pxy=rc.pixy
		Pxz=rc.pixz
		Pyy=rc.piyy
		Pyx=rc.pixy
		Pyz=rc.piyz
		Pzz=rc.pizz
		Pzx=rc.pixz
		Pzy=rc.piyz
	
	p0=(Pxx+Pyy+Pzz)/3.
	PIxx=Pxx-(p0)
	PIxy=Pxy
	PIxz=Pxz

	PIyy=Pyy-(p0)
	PIyx=Pyx
	PIyz=Pyz

	PIzz=Pzz-(p0)
	PIzx=Pzx
	PIzy=Pzy

	return PIxx,PIxy,PIxz,PIyy,PIyx,PIyz,PIzz,PIzx,PIzy, p0

def calcPID(rc,sp, dx):
	Dxx,Dxy,Dxz,Dyy,Dyx,Dyz,Dzz,Dzx,Dzy, theta=calcD(rc,sp, dx)
	PIxx,PIxy,PIxz,PIyy,PIyx,PIyz,PIzz,PIzx,PIzy, p0=calcPI(rc,sp)
	
	PI_D=PIxx*Dxx+PIxy*Dxy+PIxz*Dxz+PIyx*Dyx+PIyy*Dyy+PIyz*Dyz+PIzx*Dzx+PIzy*Dzy+PIzz*Dzz

	ptheta = p0 * theta

	return PI_D, ptheta
