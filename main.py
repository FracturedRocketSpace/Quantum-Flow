import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from scipy.stats import norm
import scipy

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
    p = np.diag(np.fft.fftfreq(len(x)),0);
    exp2 = scipy.linalg.expm(-1j*dt/2 * p**2)

    for k in range(1, len(t)):
        t1 = np.dot(exp1, psi[k-1,:]);
        t2 = np.fft.fft(t1);
        t3 = np.dot(exp2, t2);
        t4 = np.fft.ifft(t3);
        psi[k,:] = np.dot(exp1, t4);
        print(sum(psi[k,:]))

    return psi

# Define input parameters
xmin = 0;
xmax = 10.0;
dx = .01;

tmin = 0
tmax = 0.5
dt = 0.01

# Determine x and t range
x = np.arange(xmin, xmax, dx)
t = np.arange(tmin, tmax, dt)

X, T =np.meshgrid(x,t)

# Define psi at t=0
psi0= np.sin(math.pi*x)+1
rv = norm(loc = 5, scale = 1.0)
psi0 = rv.pdf(x);

# Normalize
#psi0 /= sum(psi0);

# Define potential
V = np.zeros(len(x))
V[int(len(V)/2)] = 10;

# Inititate psi
psi = np.array(np.zeros([len(t),len(x)]), dtype=np.complex128)
psi[0,:] = psi0

# Explicit method
psiExplicit = calculateExplicit(x, t, dx, dt, np.copy(psi), V);

# Suzuki/Souflaki/Gyros Method
psiGyros = calculateSuzuki(x, t, dx, dt, np.copy(psi), V);

# Plot result
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

