import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from scipy.stats import norm
import scipy
from matplotlib import cm

def calculateExplicit(x, t, dx, dt, psi, V):
    # Construct hamiltonian
    H = -1/(2 * dx**2) * (-2*np.diag(np.ones(len(x)), 0) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1))
    # Periodic boundary condition
    H[0,-1] = -1/(2 * dx**2)
    H[-1,0] = -1/(2 * dx**2)
    # Add potential
    H += np.diag(V,0)

    for k in range(0, len(t)-1):
        # Compute next time step
        psi[k+1,:] = psi[k,:] + dt/(1j) * np.dot(H,psi[k,:])

    return psi

def calculateSuzuki(x, t, dx, dt, psi, V):
    exp1 = scipy.linalg.expm(-1j*dt/2 * np.diag(V,0));

    dp = 1/(max(x)-min(x));
    p = np.linspace(-len(x)/2*dp, +len(x)/2*dp - dp, len(x) ) # Adapted from course Imaging Physics
    p = np.diag(p,0)
    exp2 = scipy.linalg.expm(-1j*dt/2 * p**2)

    for k in range(1, len(t)):
        t1 = np.dot(exp1, psi[k-1,:]);
        t2 = np.fft.fft(t1);
        t3 = np.dot(exp2, t2);
        t4 = np.fft.ifft(t3);
        psi[k,:] = np.dot(exp1, t4);
        #print(sum(psi[k,:]))

    return psi

# Define input parameters
xmin = 0;
xmax = 3*math.pi;
dx = .01;

tmin = 0
tmax = 1.0
dt = 0.01

# Determine x and t range
x = np.arange(xmin, xmax, dx)
t = np.arange(tmin, tmax, dt)

X, T =np.meshgrid(x,t)

# Define psi at t=0
psi0 = np.zeros(len(x));
psi0[int(len(x)*1/3):int(len(x)*2/3)] = np.sin( x[int(len(x)*1/3):int(len(x)*2/3)] - math.pi   )

# Define potential
V = np.zeros(len(x))
# Infite square well
#V[0:int(len(V)/3)] = 9223372036854775807
#V[int(len(V)*2/3):len(V)] = 9223372036854775807;
# Wider Well
V[0:int(len(V)/4)] = 9223372036854775807; # Max int?
V[int(len(V)*3/4):len(V)] = 9223372036854775807;

# Inititate psi
psi = np.array(np.zeros([len(t),len(x)]), dtype=np.complex128)
psi[0,:] = psi0

# Explicit method
psiExplicit = calculateExplicit(x, t, dx, dt, np.copy(psi), V);

# Suzuki/Souflaki/Gyros Method
psiGyros = calculateSuzuki(x, t, dx, dt, np.copy(psi), V);

# Plot results
fig = plt.figure(1)
ax = fig.gca(projection='3d')
wire=ax.plot_wireframe(X,T,np.real(psiExplicit),rstride=1, cstride=1)
plt.xlabel("Position")
plt.ylabel("Time")


fig = plt.figure(2)
ax = fig.gca(projection='3d')
wire=ax.plot_wireframe(X,T,np.real(psiGyros),rstride=1, cstride=1)
plt.xlabel("Position")
plt.ylabel("Time")

fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, np.real(psiGyros), rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])

plt.figure(4)
plt.plot(x,V);
plt.xlabel('Position')
plt.ylabel("Potential")

plt.figure(5)
plt.plot(x,psi0);
plt.xlabel('Position')
plt.ylabel("Initial Psy")


