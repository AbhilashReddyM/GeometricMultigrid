"""
This is an example showing how to call the mgd2d solver.
"""
import numpy as np
import time
from mgd2d import V_cycle

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#analytical solution
def Uann(x,y):
   return (x**3-x)*(y**3-y)
#RHS corresponding to above
def source(x,y):
  return 6*x*y*(x**2+ y**2 - 2)

#input
max_cycles = 50   #maximum numbera of V cycles
nlevels    = 8    #number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 1*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 1*2**(nlevels-1) #
tol        = 1e-10      

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
u   =np.zeros([NX+2,NY+2])#approximation
f   =np.zeros([NX+2,NY+2])#RHS

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY)
f[1:NX+1,1:NY+1]   =source(XX,YY)

print('mgd2d.py solver:')
print('NX:',NX,', NY:',NY,', tol:',tol,'levels: ',nlevels)

#start solving
tb=time.time()

##V cycle
for it in range(1,max_cycles+1):
  u,res=V_cycle(NX,NY,nlevels,u,f)
  rtol=np.max(np.max(np.abs(res)))
  if(rtol<tol):
    break
  error=uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1]
  print('  cycle: ',it,', L_inf(res.)= ',rtol,',L_inf(true error): ',np.max(np.max(np.abs(error))))

print('Elapsed time: ',time.time()-tb,' seconds')

error=uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1]
print('L_inf (true error): ',np.max(np.max(np.abs(error))))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, u[1:NX+1,1:NY+1],cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()


