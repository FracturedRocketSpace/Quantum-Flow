import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'!
from scipy.stats import norm
import scipy
from matplotlib import cm
from matplotlib import animation
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
        #Normalize new psi        
        Norm=scipy.integrate.trapz(np.absolute(psi[k+1,:])**2,dx=dx)
        psi[k+1,:]*=1/(Norm**(1/2))
    
    print("Done",flush=True)
    return psi   
    
def calculateImplicit(x,t,dx,dt,psi,V):
    print("Calculating implicit method...",flush=True)
    #Construct hamiltonian
    diagonals=[-2*np.ones(len(x)), np.ones(len(x)-1), np.ones(len(x)-1)] 
    H = -1/(2 * dx**2) * scipy.sparse.diags(diagonals, [0,1,-1], format="csc")
    H += scipy.sparse.diags(V,0,format="csc")    
    
    # Periodic boundary condition
    H[0,-1] = -1/(2 * dx**2)
    H[-1,0] = -1/(2 * dx**2)    
    
    #H2 = 1j/dt*scipy.sparse.identity(len(x), format="csc") - H
    H2 = scipy.sparse.identity(len(x),format="csc") + 1j*dt*H/2    
    
    #Invert hamiltonian    
    #H2Inv=scipy.sparse.linalg.inv(H2)    
    
    #LU factorization of Hamiltonian
    factors = scipy.sparse.linalg.factorized(H2)

    #Compute next time step
    for k in range(0, len(t)-1):
        #psi[k+1,:] = H2Inv.dot(1j/dt*psi[k,:])
        #psi[k+1,:] = factors(1j/dt*psi[k,:])
        psi[k+1,:] = factors(psi[k,:])
        #Normalize new psi        
        Norm=scipy.integrate.trapz(np.absolute(psi[k+1,:])**2,dx=dx)
        psi[k+1,:]*=1/(Norm**(1/2))
        
    print("Done",flush=True)  
    return psi
    
def calculateCrank(x,t,dx,dt,psi,V):
    print("Calculating Crank method...", flush=True)
    #Construct hamiltonian
    diagonals=[-2*np.ones(len(x)), np.ones(len(x)-1), np.ones(len(x)-1)] 
    H = -1/(2 * dx**2) * scipy.sparse.diags(diagonals, [0,1,-1], format="csc")
    H += scipy.sparse.diags(V,0,format="csc")    
    
    # Periodic boundary condition
    H[0,-1] = -1/(2 * dx**2)
    H[-1,0] = -1/(2 * dx**2)
    
    H2 = scipy.sparse.identity(len(x), format="csc") - 1j*dt*H/2
    H3 = scipy.sparse.identity(len(x), format="csc") + 1j*dt*H/2
    
    #LU Factorization
    factors = scipy.sparse.linalg.factorized(H3)
    #inverseDenominator = factors(np.identity(len(x)))    
    #operator = H2.dot(inverseDenominator)    
    
    #Hamiltonian inversion
    #inverseDenominator = scipy.sparse.linalg.inv(scipy.sparse.identity(len(x), format="csc")+1j*dt*H/2)
    #operator = (scipy.sparse.identity(len(x), format="csc")-1j*dt*H/2).dot(inverseDenominator)    
    
    #Compute next time step
    for k in range(0, len(t)-1):
        psi[k+1,:] = factors(H2.dot(psi[k,:]))
        #psi[k+1,:] = operator.dot(psi[k,:])
   
    print("Done", flush=True)    
    
    return psi
    
def calculateSuzuki(x, t, dx, dt, psi, V):
    print("Calculating FT method...",flush=True)
    # Calculate momenta
    # Can someone find why this is approximately correct?
    p = 2/3 * np.fft.fftfreq(len(x)) * len(x); # I would have expected dp here. But then it oscilates much more slowly than other method

    # Calculate exponents
    exp1 = scipy.sparse.diags(np.exp(-1j*dt/2*V),0,format="csc")
    exp2 = scipy.sparse.diags(np.exp(-1j*dt/2* (p)**2 ),0,format="csc")

    for k in range(1, len(t)):
        t1 =exp1.dot(psi[k-1])
        t2 = np.fft.fft(t1);
        t3 = exp2.dot(t2);
        t4 = np.fft.ifft(t3);
        psi[k,:] = exp1.dot(t4)

    print("Done",flush=True)
    return psi
        
    
# Define input parameters
xmin = 0;
xmax = 3*math.pi;
dx = .01;

tmin = 0
tmax = 0.5
dt =  0.0001

# Determine x and t range
x = np.arange(xmin, xmax, dx)
t = np.arange(tmin, tmax, dt)

X, T =np.meshgrid(x,t)

# Define psi at t=0
packetWidthSqr = 1/16
packetCenter = 3/8*xmax
k = 40
psi0 = np.exp(-(x-packetCenter)**2/(2*packetWidthSqr)+1j*k*x)
#Normalize psi
Norm=scipy.integrate.trapz(np.absolute(psi0)**2,dx=dx)
psi0*=1/(Norm**(1/2))

# Choose the potential
def definePotential(choosePotential):
    # Define potential
    V = np.zeros(len(x))
    error = False;
    
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
    elif (choosePotential == "BAR"):
        V[int(len(V)*5/8):int(len(V)*6/8)] = 300
    else: 
        error=True
        
    return V, error

# Prompt user for potential
print("Choose one of the following potentials \n",
      "ISW = Infinite Square Well \n",
      "WW = Wider Well\n",
      "DAP = Double aperture potential\n",
      "BAR = Barrier",
      flush=True)
choosePotential = input("Input:")

V, error = definePotential(choosePotential)

if error:
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
surf = ax.plot_surface(X, T, np.absolute(psiImplicit), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
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
surf = ax.plot_surface(X, T, np.absolute(psiCrank), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.view_init(90, 90); # Top view
plt.xlabel("Position")
plt.ylabel("Time")
plt.title("Crank method")
fig.colorbar(surf, shrink=0.5, aspect=5)
# Hide z-axis
ax.w_zaxis.line.set_lw(0.)
ax.set_zticks([])

fig = plt.figure(4)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, T, np.absolute(psiGyros), rstride=10, cstride=10,cmap=cm.coolwarm, linewidth=0, antialiased=False)
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

# Anitmated plot of Psi
fig = plt.figure()
ax = plt.axes(xlim=(xmin,xmax), ylim=(np.min(np.real(psiCrank)),np.max(np.real(psiCrank))))
line, = ax.plot([],[], lw=2)
axPot = ax.twinx()
pot = axPot.plot([],[], lw=1)
timestamp = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def animate(i):
    line.set_data(x, np.absolute(psiCrank[i,:]))
    timestamp.set_text("Time = %.3f" % t[i] )
    return line, timestamp

anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=10, blit=True)

axPot.plot(x, V, 'r')

ax.set_xlabel("Position")
ax.set_ylabel("Wave function", color='b')
axPot.set_ylabel("Potential", color='r')
plt.title("Animated Crank method wave function")
plt.show()
