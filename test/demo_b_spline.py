import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y, z = numpy arrays of your peak coordinates
spl = SmoothBivariateSpline(x, y, z, kx=3, ky=3, s=1e4)   # tune s
gx, gy = np.meshgrid(
    np.linspace(x.min(), x.max(), 400),
    np.linspace(y.min(), y.max(), 400),
    indexing='xy'
)
gz = spl.ev(gx.ravel(), gy.ravel()).reshape(gx.shape)

fig = plt.figure(figsize=(9,6))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=4, c='k', alpha=0.25)
ax.plot_surface(gx, gy, gz, rstride=8, cstride=8, alpha=0.5)
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.tight_layout(); plt.show()
