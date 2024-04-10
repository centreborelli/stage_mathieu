
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
	s = nanquantile(fabs(x), q)       # find saturation quantile
	r = 1 - clip(x/s, 0, 1)        # red component
	g = 1 - clip(fabs(x/s), 0, 1)  # green
	b = 1 + clip(x/s, -1, 0)       # blue
	c = dstack([r, g, b])          # color
	c = clip(c, 0, 1)              # saturate color into [0,1]
	c = nan_to_num(c, nan=0.8)
	c = (255*c).astype(int)        # rescale and quantize
	return c


# auxiliary function to compute the Gaussian blur of an image
def blur_gaussian(x, σ):
	from numpy.fft import fft2, ifft2, fftfreq
	from numpy import meshgrid, exp
	h,w = x.shape
	p,q = meshgrid(fftfreq(w), fftfreq(h))
	X = fft2(x)
	F = exp(-σ**2 * (p**2 + q**2))
	Y = F*X
	y = ifft2(Y).real
	return y

import iio
x = iio.read("http://gabarro.org/img/gbarbara.png").squeeze()
y = blur_gaussian(x, 10)

iio.gallery([x,sauto(laplacian(y))])


def bessel_dirichlet(n, m, x):
	"""
	Compute the nth Bessel function of x scaled
	to Dirichlet boundary conditions at the mth root
	"""
	from scipy.special import jv, jn_zeros

	z = jn_zeros(n, m+1)[m]  # mth zero of J_n
	y = jv(n, x*z)           # J_n scaled between 0 and z
	return y


def disk_dirichlet(n, m, k, s, r, Θ):
	"""
	Disk eigenmodes
	n : order of bessel function
	m : root of nth bessel function
	k : number of oscillations in angular part
	s : sign of the oscillations (even/odd)
	r : input array with r-coordinates
	θ : input array with θ-coordinates
	"""
	from numpy import sin,cos 
	a = bessel_dirichlet(n, m, r)
	f = sin if s%2 else cos
	b = f(k*θ)
	return a * b


def polar(x, y):
	""" polar from euclidean coordinates """
	from numpy import arctan2, nan
	r = (x**2 + y**2)**0.5
	θ = arctan2(y, x)
	θ[r>1] = nan
	r[r>1] = nan
	return r,θ


def canvas(N):
	from numpy import linspace, meshgrid
	x = linspace(-1.1, 1.1, N)
	y = linspace(-1.1, 1.1, N)
	X,Y = meshgrid(x,y)
	return X,Y

x,y = canvas(200)
r,θ = polar(x,y)

iio.gallery([sauto(x),sauto(y),sauto(r),sauto(θ)])

j_3_7 = bessel_dirichlet(13, 2, r)

from scipy.special import jv

iio.gallery([sauto(jv(0,20*r)),sauto(jv(1,20*r)),sauto(jv(2,20*r)),sauto(jv(3,20*r)),sauto(jv(13,20*r))])

d = disk_dirichlet

iio.gallery([sauto(d(10,2,6,0,r,θ)+d(10,2,6,1,r,θ))])


