# # Disk eigenmodes
#
# In this notebook we perform some experiments with the eigenfunctions of a
# circular domain.  More precisely, we consider the Helmholtz equation
# $$\begin{cases}\Delta u=-\lambda u&\Omega\\u=0&\partial\Omega\end{cases}$$
# on the unit disk $\Omega=\{x\in\mathbf{R}^2\ :\ \|x\|\le 1\}$.  In polar
# coordinates, solutions of this equation are of the form
# $$u(r,\theta)=f(k\theta)g(r)$$
# where $f=\sin$ or $f=\cos$ and $g(r)=J_n(j_{n,m}r)$, with $j_{n,m}$ being the
# $m$th root of the $n$th bessel function $J_n$.  Notice that, algthough all
# solutions are of this form, not all choices of $(k,n,m)\in\mathbf{Z}^3$
# result in actual solutions.  As we will see below, we must have $m=k+1$.


# ## 0. Auxiliary functions
#
# Let use define some utility functions that will be used below.

# ! python -m pip install numpy scipy imgra iio

# auxiliary function to compute the L2 inner product between two images
def inner_product(x, y):
	return x.flatten().T @ y.flatten()

# auxiliary function to compute the laplacian of an image
def laplacian(x):
	""" Compute the five-point laplacian of an image """
	import imgra                  # image processing with graphs
	s = x.shape                   # shape of the domain
	B = imgra.grid_incidence(*s)  # discrete gradient operator
	L = -B.T @ B                  # laplacian operator
	y = L @ x.flatten()           # laplacian of flattened data
	return y.reshape(*s)          # reshape and return

# auxiliary function to show a signed image with red-white-blue palette
def sauto(x, q=0.995):
	""" RGB rendering of a signed scalar image using a divergent palette """
	from numpy import clip, fabs, dstack, nanquantile, nan_to_num
	s = nanquantile(fabs(x), q)    # find saturation quantile
	r = 1 - clip(x/s, 0, 1)        # red component
	g = 1 - clip(fabs(x/s), 0, 1)  # green
	b = 1 + clip(x/s, -1, 0)       # blue
	c = dstack([r, g, b])          # color
	c = clip(c, 0, 1)              # saturate color into [0,1]
	c = nan_to_num(c, nan=0.5)     # set nans to gray
	c = (255*c).astype(int)        # rescale and quantize
	return c

# auxiliary function to compute the Gaussian blur of an image
def blur_gaussian(x, σ):
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid, exp
	h,w = x.shape                           # shape of the rectangle
	p,q = meshgrid(fftfreq(w), fftfreq(h))  # build frequency abscissae
	X = fft2(x)                             # move to frequency domain
	F = exp(-σ**2 * (p**2 + q**2))          # define filter
	Y = F*X                                 # apply filter
	y = ifft2(Y).real                       # go back to spatial domain
	return y
#import iio
#x = iio.read("http://gabarro.org/img/gbarbara.png").squeeze()
#y = blur_gaussian(x, 10)
#iio.gallery([x,sauto(y),sauto(laplacian(y))])


# ## 1. Polar coordinates

def polar(x, y):
	""" polar from euclidean coordinates """
	from numpy import arctan2, nan
	r = (x**2 + y**2)**0.5
	θ = arctan2(y, x)
	θ[r>1] = nan
	r[r>1] = nan
	return r,θ


def canvas(N):
	""" define a grid of size NxN centered around (0,0) """
	from numpy import linspace, meshgrid
	x = linspace(-1.1, 1.1, N)
	y = linspace(-1.1, 1.1, N)
	X,Y = meshgrid(x,y)
	return X,Y


def test_polarcoords():
	x,y = canvas(200)
	r,θ = polar(x,y)
	import iio
	iio.gallery([sauto(x),sauto(y),sauto(r),sauto(θ)])
test_polarcoords()


# ## 2. Separable functions
#
# The core element are the bessel functions $J_n$, whose argument has been
# linearly scaled so that their $m$th root of $J_n$ happens at abcissa 1.
# The implementation relies on the functions `jv` and `jn_zeros` from
# `scipy.special`

def bessel_dirichlet(n, m, x):
	"""
	Compute the nth Bessel function of x scaled
	to Dirichlet boundary conditions at the mth root
	"""
	from scipy.special import jv, jn_zeros
	z = jn_zeros(n, m+1)[m]  # mth zero of J_n
	y = jv(n, x*z)           # J_n scaled between 0 and z
	return y


def disk_dirichlet(n, m, k, s, r, θ):
	"""
	Disk eigenmodes
	n : order of bessel function
	m : root of nth bessel function
	k : number of oscillations in angular part
	s : sign of the oscillations (0=cos, 1=sin)
	r : input array with r-coordinates
	θ : input array with θ-coordinates
	"""
	from numpy import sin,cos
	a = bessel_dirichlet(n, m, r)
	f = sin if s%2 else cos
	b = f(k*θ)
	return a * b

def test_angular():
	x,y = canvas(200)
	r,θ = polar(x,y)
	def d(n,m,k,s):
		X = disk_dirichlet(n,m,k,s,r,θ)
		return X
		#return sauto(laplacian(X)/X)
	import iio
	from numpy import hstack,vstack
	X = hstack([d(i,2,i,0) for i in range(4)])
	LX = laplacian(X)
	Y = vstack([sauto(X), sauto(LX), sauto(LX/X)])
	iio.display(Y)
test_angular()


def test_radial():
	x,y = canvas(200)
	r,θ = polar(x,y)
	def d(n,m,k,s):
		X = disk_dirichlet(n,m,k,s,r,θ)
		return X
		#return sauto(laplacian(X)/X)
	import iio
	from numpy import hstack,vstack
	X = hstack([d(0,i,0,0) for i in range(4)])
	LX = laplacian(X)
	Y = vstack([sauto(X), sauto(LX), sauto(LX/X)])
	iio.display(Y)
test_radial()

# ## 3. Construction of the eigenmodes
#
# The core element are the bessel functions $J_n$, whose argument has been
# linearly scaled so that their $m$th root of $J_n$ happens at abcissa 1.
# The implementation relies on the functions `jv` and `jn_zeros` from
# `scipy.special`


# ## 4. Verification of Helmholtz equation

# ## 5. Verification of orthogonality

# ## 6. Verification of span

# ## 7. Construction of eigenvalues

# ## 8. The Neumann case

# ## 9. Verification of Pólya conjecture


# def f(x):
# 	π = x
# 	print(f"π = {π}")
# 	def g(y):
# 		z = y + π
# 		print(f"z = {z}")
# 	g(x)
# f(5)









x,y = canvas(200)
r,θ = polar(x,y)

import iio

j_3_7 = bessel_dirichlet(13, 2, r)

from scipy.special import jv

iio.gallery([sauto(jv(0,20*r)),sauto(jv(1,20*r)),sauto(jv(2,20*r)),sauto(jv(3,20*r)),sauto(jv(13,20*r))])

d = disk_dirichlet

iio.gallery([sauto(d(10,2,6,0,r,θ)),sauto(d(10,2,6,1,r,θ))])


def D(n,m,s):
	return sauto(disk_dirichlet(n,m,n,s,r,θ))


def C(n,m): return disk_dirichlet(n,m,n,0,r,θ)
def S(n,m): return disk_dirichlet(n,m,n,1,r,θ)


iio.gallery([D(0,0,0),D(1,0,0),D(2,0,0),D(3,0,0),D(13,0,0)])

import numpy
a = D(0,0,0)
b = D(0,1,0)
a.shape
iio.display(numpy.vstack([numpy.hstack([D(i,j,1) for i in range(0,4)]) for j in range(0,5)]))
iio.display(numpy.vstack([numpy.hstack([D(i,j,0) for i in range(0,4)]) for j in range(0,5)]))

iio.gallery([D(0,0,0),D(0,1,0),D(0,2,0),D(0,3,0),D(0,13,0)])

iio.gallery([D(1,0,0),D(1,1,0),D(1,2,0),D(1,3,0),D(1,13,0)])

iio.gallery([D(0,1,0),D(1,1,0),D(2,1,0),D(3,1,0),D(13,1,0)])

iio.gallery([D(1,1,0),D(2,2,0),D(3,3,0),D(4,4,0),D(13,13,0)])

import numpy
numpy.nansum(C(3,1)*C(3,2))

iio.gallery([sauto(C(3,15)/laplacian(C(3,15)))])




