import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from scipy.stats import norm
import scipy
from matplotlib import cm
import sys

def calculateExplicit(x, t, dx, dt, psi, V):
    print("Calculating explicit method...",flush=True)
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
    
    print("Done",flush=True)
    return psi   
    
def calculateImplicit(x,t,dx,dt,psi,V):
    print("Calculating implicit method...",flush=True)
    #Construct hamiltonian
    diagonals=[-2*np.ones(len(x)), np.ones(len(x)-1), np.ones(len(x)-1)] 
    H = -1/(2 * dx**2) * scipy.sparse.diags(diagonals, [0,1,-1], format="csc")
    H += scipy.sparse.diags(V,0,format="csc")    
    H2 = 1j/dt*scipy.sparse.identity(len(x), format="csc") - H
    
    #Invert hamiltonian    
    H2Inv=scipy.sparse.linalg.inv(H2)    
    
    #Compute next time step
    for k in range(0, len(t)-1):
        psi[k+1,:] = H2Inv.dot(1j/dt*psi[k,:])
    
    print("Done",flush=True)  
    return psi
    
def calculateCrank(x,t,dx,dt,psi,V):
    #Construct hamiltonian
    diagonals=[-2*np.ones(len(x)), np.ones(len(x)-1), np.ones(len(x)-1)] 
    H = -1/(2 * dx**2) * scipy.sparse.diags(diagonals, [0,1,-1], format="csc")
    H += scipy.sparse.diags(V,0,format="csc")    
    inverseDenominator = scipy.sparse.linalg.inv(scipy.sparse.identity(len(x), format="csc")+1j*dt*H/2)
    operator = (scipy.sparse.identity(len(x), format="csc")-1j*dt*H/2).dot(inverseDenominator)
    
    operatorInv = scipy.sparse.linalg.inv(operator)

    #Compute next time step
    for k in range(0, len(t)-1):
        psi[k+1,:] = operatorInv.dot(psi[k,:])
        
    return psi
    

def calculateSuzuki(x, t, dx, dt, psi, V):
    print("Calculating FT method...",flush=True)    
    
    exp1 = scipy.linalg.expm(-1j*dt/2 * np.diag(V,0));

    p = np.fft.fftfreq(len(x)) * len(x); # I would have expected dp here. But then it oscilates much more slowly than other method
    
    L = (max(x)-min(x))
    dp = 2*np.pi/L;    
    #temp = np.fft.ifftshift(np.linspace(-len(x)/2,len(x)/2,len(x)))
    #p = temp * dp;
    
    p = np.diag(p,0)
    exp2 = scipy.linalg.expm(-1j*dt/2 * p**2)

    for k in range(1, len(t)):
        t1 = np.dot(exp1, psi[k-1,:]);
        t2 = np.fft.fft(t1);
        t3 = np.dot(exp2, t2);
        t4 = np.fft.ifft(t3);
        psi[k,:] = np.dot(exp1, t4);
        #print(sum(psi[k,:]))
    
    print("Done",flush=True)
    return psi
        
    
# Define input parameters
xmin = 0;
xmax = 3*math.pi;
dx = .01;

tmin = 0
tmax = 25
dt = 1

# Determine x and t range
x = np.arange(xmin, xmax, dx)
t = np.arange(tmin, tmax, dt)

X, T =np.meshgrid(x,t)

# Define psi at t=0
psi0 = np.zeros(len(x));
psi0[int(len(x)*1/3):int(len(x)*2/3)] = np.sin( x[int(len(x)*1/3):int(len(x)*2/3)] - math.pi   )
#Normalize psi
Norm=scipy.integrate.trapz(psi0**2,dx=dx)
psi0*=1/(Norm**(1/2))

# Choose the potential
def definePotential(choosePotential):
    # Define potential
    V = np.zeros(len(x))
    error = 0;
    
    if (choosePotential == "ISW"):
        # Infite square well
        V[0:int(len(V)/3)] = 9223372036854775807
        V[int(len(V)*2/3):len(V)] = 9223372036854775807;
    elif (choosePotential == "WW"):
        # Wider Well
        V[0:int(len(V)/4)] = 9223372036854775807; # Max int?
        V[int(len(V)*3/4):len(V)] = 9223372036854775807;
    elif (choosePotential == "DAP"):
        # Double aperture potential
        a=0.3; #distance between apertures
        d=0.1; #diameter of apertures
        V[0:int((0.5-a/2-d/2)*len(V))] = 9223372036854775807
        V[int((0.5-a/2+d/2)*len(V)):int((0.5+a/2-d/2)*len(V))]=9223372036854775807
        V[int((0.5+a/2+d/2)*len(V)):-1]=9223372036854775807
    else: 
        error=1
#        sys.exit("Unknown potential")
#        quit()
    
    return V, error

print("Choose one of the following potentials \n",
      "ISW = Infinite Square Well \n",
      "WW = Wider Well\n",
      "DAP = Double aperture potential\n",
      flush=True)
choosePotential = input("Input:")

V, error = definePotential(choosePotential)

if error==1:
    sys.exit("Unknown potential")
    
# Inititate psi
psi = np.array(np.zeros([len(t),len(x)]), dtype=np.complex128)
psi[0,:] = psi0

# Explicit method
#psiExplicit = calculateExplicit(x, t, dx, dt, np.copy(psi), V);

#Implicit mehod
psiImplicit = calculateImplicit(x, t, dx, dt, np.copy(psi), V)

# Suzuki/Souflaki/Gyros Method
psiGyros = calculateSuzuki(x, t, dx, dt, np.copy(psi), V);

# Crank–Nicholson Method
psiCrank = calculateCrank(x, t, dx, dt, np.copy(psi), V);

# Plot results
#fig = plt.figure(1)
#ax = fig.gca(projection='3d')
#wire=ax.plot_wireframe(X,T,np.real(psiExplicit),rstride=1, cstride=1)
#plt.xlabel("Position")
#plt.ylabel("Time")


#fig = plt.figure(2)
#ax = fig.gca(projection='3d')
#wire=ax.plot_wireframe(X,T,np.real(psiGyros),rstride=1, cstride=1)
#plt.xlabel("Position")
#plt.ylabel("Time")

fig = plt.figure(2)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, np.real(psiImplicit), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
plt.title("Implicit method")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])

fig = plt.figure(3)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, np.real(psiCrank), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])

fig = plt.figure(4)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, np.real(psiGyros), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
plt.title("FT method")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])


plt.figure(5)
plt.plot(x,V);
plt.xlabel('Position')
plt.ylabel("Potential")

plt.figure(6)
plt.plot(x,psi0);
plt.xlabel('Position')
plt.ylabel("Initial Psi")