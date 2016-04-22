import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from scipy.stats import norm
from matplotlib import animation
import scipy
from matplotlib import cm

# Define input parameters
xmin = 0;
xmax = 2*math.pi;
dx = 0.05;

ymin = 0
ymax = 2*math.pi;
dy = 0.05;

tmin = 0
tmax = 0.05
dt = 0.0002

# Determine x and t range
x = np.arange(xmin, xmax, dx)
y = np.arange(ymin, ymax, dy)
t = np.arange(tmin, tmax, dt)

X, Y =np.meshgrid(x,y)

print('Init Psy and potential')
# Define psi at t=0
V = np.zeros((len(x),len(y)));
for k in range(0, len(y)):
    if(k < int(len(x)*1/4) or k > int(len(x)*3/4) ):
        V[:,k] = 10**20;
    else:
        V[0:int(len(x)*1/4),k] = 10**20
        V[int(len(x)*3/4):-1,k] = 10**20
V = np.squeeze(np.reshape(V,(len(x)*len(y),1)))

psi0 = np.zeros((len(x),len(y)));
for k in range(int(len(y)*1/4), int(len(y)*3/4) ):
    psi0[int(len(x)*1/4):int(len(x)*3/4),k] = np.sin( x[int(len(x)*1/4):int(len(x)*3/4)] - math.pi/2   ) * np.sin( y[k] - math.pi/2   )
psi0 = np.squeeze(np.reshape(psi0,(len(x)*len(y),1)))

# Inititate psi
psi = np.array(np.zeros([len(t),len(x)*len(y)]), dtype=np.complex128)
psi[0,:] = psi0
psi2 = np.copy(psi);

## Create Hamiltonian
print('Constructing Hamiltonians')
# x at -1 0 1
x_diag = [-2*np.ones(len(x)*len(y)), np.ones(len(x)*len(y))-1, np.ones(len(x)*len(y))-1,]
# y at -nx, 0, nx
y_diag =  [-2*np.ones(len(x)*len(y)), np.ones(len(x)*len(y))-len(x), np.ones(len(x)*len(y))-len(x),]
# Combine with potential
H = -1/(2*dx**2) * scipy.sparse.diags(x_diag, [0, 1, -1], format="csc")
H += -1/(2*dy**2) * scipy.sparse.diags(x_diag, [0,len(x), -len(x),], format="csc");
H += scipy.sparse.diags(V,0,format="csc");

## Construct operators
print('Constructing operators')
Op = scipy.sparse.identity(len(x)*len(y), format="csc") + 1j*dt*H;

Op1 = scipy.sparse.identity(len(x)*len(y), format="csc") - 1j*dt*H/2;
Op2 = scipy.sparse.identity(len(x)*len(y), format="csc") + 1j*dt*H/2;
OpCrank = Op1.dot(scipy.sparse.linalg.inv(Op2))

## Invert
print('Inverting')
OpInv = scipy.sparse.linalg.inv(Op);

#Compute next time step
print('Starting loop')
for k in range(0, len(t)-1):
    psi[k+1,:] = OpInv.dot(psi[k,:])
    psi2[k+1,:] = OpCrank.dot(psi2[k,:])

# Plot

#
fig =plt.figure(1)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, np.real(np.reshape(psi0,(len(y),len(x)), order='F' )), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position x")
plt.ylabel("Position y")
plt.title("Start")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])
#
fig =plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, np.real(np.reshape(V,(len(y),len(x)), order='F')), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position x")
plt.ylabel("Position y")
plt.title("Potential")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])

# Plot
sync_num = np.zeros(2);
#
plot_args = {'cmap':cm.coolwarm, 'linewidth':0, 'antialiased':False, 'vmin':-1, 'vmax':1}
fig1 = plt.figure(3)
ax1 = fig1.gca(projection='3d')
ax1.view_init(90, 90); # Top view
surf1 = ax1.plot_surface(X, Y, np.real(np.reshape(psi0,(len(y),len(x)), order='F')) , **plot_args  )
fig1.colorbar(surf1, shrink=0.5, aspect=5)

def update_3d1(num):
    # Plot 1
    ax1.clear()
    surf1 = ax1.plot_surface(X, Y, np.real(np.reshape(psi[num],(len(y),len(x)), order='F')) , **plot_args  )
    #
    ax1.set_xlabel("Position X")
    ax1.set_ylabel("Position Y")
    ax1.set_title("Implicit: Time = %.4f" % t[num])
    ax1.set_zlim(-1, 1)
    # Hide z-axis
    ax1.w_zaxis.line.set_lw(0.)
    ax1.set_zticks([])
    sync_num[0]=num;
    #
    return surf1

line_ani = animation.FuncAnimation(fig, update_3d1, frames=len(t), interval=100, blit=False)
##
fig2 = plt.figure(4)
ax2 = fig2.gca(projection='3d')
ax2.view_init(90, 90); # Top view
surf2 = ax2.plot_surface(X, Y, np.real(np.reshape(psi0,(len(y),len(x)), order='F')) , **plot_args  )
fig2.colorbar(surf2, shrink=0.5, aspect=5)

def update_3d2(num):
    num=sync_num[0];
    # Plot 1
    ax2.clear()
    surf2 = ax2.plot_surface(X, Y, np.real(np.reshape(psi2[num],(len(y),len(x)), order='F')) , **plot_args  )
    #
    ax2.set_xlabel("Position X")
    ax2.set_ylabel("Position Y")
    ax2.set_title("Crank: Time = %.4f" % t[num])
    ax2.set_zlim(-1, 1)
    # Hide z-axis
    ax2.w_zaxis.line.set_lw(0.)
    ax2.set_zticks([])
    #
    return surf2

line_ani2 = animation.FuncAnimation(fig, update_3d2, frames=len(t), interval=100, blit=False)