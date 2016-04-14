import numpy as np
import math
import matplotlib.pyplot as plt


# Define input parameters
m = 1;
hbar = 1;

xmin = 0;
xmax = 1;
dx = .01;

tmin = 0
tmax = 1
dt = 0.01

# Determine x and t range
x = np.arange(xmin, xmax, dx)
t = np.arange(tmin, tmax, dt)

X, T =np.meshgrid(x,t)

# Define psi at t=0
psi0= np.sin(math.pi*x)+1

# Define potential
V = np.zeros(len(x))

# Inititate psi
psi = np.array(np.zeros([len(t),len(x)]), dtype=np.complex128)
psi[0,:] = psi0

# Explicit method

# Construct hamiltonian
H = -hbar**2/(2 * m * dx**2) * (-2*np.diag(np.ones(len(x)), 0) + np.diag(np.ones(len(x)-1), 1) + np.diag(np.ones(len(x)-1), -1))
# Periodic boundary condition
H[0,-1] = -hbar**2/(2 * m * dx**2)
H[-1,0] = -hbar**2/(2 * m * dx**2)
# Add potential
H += np.diag(V,0)

for k in range(0, len(t)-1):
    # Compute next time step
    psi[k+1,:] = psi[k,:] + dt/(1j*hbar) * np.dot(H,psi[k,:])

# Plot result
fig = plt.figure(1)
ax = fig.gca(projection='3d')
wire=ax.plot_wireframe(X,T,np.real(psi),rstride=1, cstride=1)
plt.xlabel("Position")
plt.ylabel("Time")

