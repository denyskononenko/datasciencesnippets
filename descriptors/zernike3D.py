import numpy as np

from scipy.special import sph_harm
from math import factorial as fact
from math import pi, sqrt, sin, cos, atan2
from numpy.linalg import norm

import matplotlib as mpl
import matplotlib.cm as cm

def binomial(n, k): return fact(n)  / (fact(k) * fact(n - k))

class Zernike3D:
    def __init__(self, ds: float):
        self.ds = ds 

    def radial(self, n: int, l: int) -> ():
        """Radial Zernike polynomials normalized for 3D case for given order n, l."""
        # normalization factor
        Q = lambda k, l, nu: ((-1)**(k + nu) / 4**k) *\
                             sqrt((2 * l + 4 * k + 3) / 3) *\
                             (binomial(2 * k, k) * binomial(k, nu) * binomial(2 * (k + l + nu) + 1, 2 * k) / binomial(k + l + nu, k))
        if (n - l) % 2 != 0: 
            return lambda r: 0
        else:
            return lambda r: sum([Q((n - l) / 2, l, nu) * r**(2 * nu + l) for nu in range((n - l) // 2 + 1)]) 
    
    def zfunction(self, n: int, l: int, m: int) -> ():
        """Zernike 3D function of given order n, l, m."""
        # check parameters validity
        # init radial part
        R = self.radial(abs(n), abs(l))
        # 3D Zernike function
        def Z(x, y, z) -> ():
            if x**2 + y**2 + z**2 > 1:
                return 0
            else:
                r = sqrt(x**2 + y**2 + z**2)
                theta = atan2(sqrt(x**2 + y**2), z)
                phi = atan2(y, x)
                return  R(r) * sph_harm(m, l, phi, theta)
        return Z

    def zfunction_on_grid(self, n: int, l: int, m: int) -> np.array:
        """Calculate 3D Zernike function on the grid with shape of the given 3D image."""
        # discretization parameters
        s = np.arange(-1, 1 + self.ds, self.ds)
        Z = self.zfunction(n, l, m)
        Z_on_grid = np.array([[[np.real(Z(x, y, z)) for x in s] for y in s] for z in s])
        return Z_on_grid

    def zfunction_point_cloud(self, n: int, l: int, m: int) -> np.array:
        """Calculate 3D Zernike function in the form of point cloud."""
        # make point cloud vertices
        s = np.arange(-1, 1 + self.ds, self.ds)
        Z = self.zfunction(n, l, m)
        Z_point_cloud = np.array([[x, y, z, np.real(Z(x, y, z))] for x in s for y in s for z in s if x**2 + y**2 + z**2 <= 1])
        # coloring setup
        norm = mpl.colors.Normalize(vmin=np.min(Z_point_cloud[:,3]), vmax=np.max(Z_point_cloud[:,3]))
        cmap = cm.winter
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        # add color to the point cloud 
        # return format: x y z r ,g, b, Z_value
        Z_point_cloud_colored = np.array([[x, y, z, m.to_rgba(Z_v)[0], m.to_rgba(Z_v)[1], m.to_rgba(Z_v)[2], Z_v] for x, y, z, Z_v in Z_point_cloud])
        
        return Z_point_cloud_colored

    def allowed_modes(self, nmax: int) -> list:
        """Generate list of (n, l, m) allowed modes for given maximum order of n: nmax."""
        return [(n, l, m) for n in range(nmax + 1) for l in range(nmax + 1) for m in range(-l, l+1) if (n - l) % 2 == 0 and n >= l]

    def save_zfunction_point_cloud(self, n: int, l: int, m: int, pathtosave: str):
        """Save point cloud for given Zernike 3D function."""
        Z_point_cloud_colored = self.zfunction_point_cloud(n, l, m)
        np.save(f"{pathtosave}/{n}_{l}_{m}.npy", Z_point_cloud_colored)
