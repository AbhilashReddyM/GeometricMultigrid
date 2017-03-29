"""
This is an example showing how to call the mgd2d solver.
V cycle and Full Multigrid are demonstrated
"""
import numpy as np
import time
from mgd3d import V_cycle,FMG,GSrelax

#analytical solution
def Uann(x,y,z,n):
  return np.sin(2*n*np.pi*x)*np.sin(2*n*np.pi*y)*np.sin(2*n*np.pi*z)

#RHS corresponding to above
def source(x,y,z,n):
  return -12 * (np.pi)**2 * n**2 * np.sin(2*n*np.pi*x) * np.sin(2*n*np.pi*y) * np.sin(2*n*np.pi*z)

#input
max_cycles = 20           #maximum number of V cycles
nlevels    = 5            #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 3*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 3*2**(nlevels-1) 
NZ         = 3*2**(nlevels-1) 
tol        = 1e-6

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2,NZ+2])#analytical solution
u   =np.zeros([NX+2,NY+2,NZ+2])#approximation
f   =np.zeros([NX+2,NY+2,NZ+2])#RHS

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY
DZ=1.0/NZ

xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
zc=np.linspace(0.5*DZ,1-0.5*DZ,NZ)

XX,YY,ZZ=np.meshgrid(xc,yc,zc)

uann[1:NX+1,1:NY+1,1:NZ+1] = Uann  (XX,YY,ZZ,1)
f   [1:NX+1,1:NY+1,1:NZ+1] = source(XX,YY,ZZ,1)

print('mgd3d.py solver:')
print('NX:',NX,', NY:',NY,'NZ:',NZ,', tol:',tol,'levels: ',nlevels)

#start solving
tb=time.time()

#V cycle
for it in range(1,max_cycles+1):
  u,res=V_cycle(NX,NY,NZ,nlevels,u,f)
  rtol=np.max(np.max(np.abs(res)))
  if(rtol<tol):
    break
  error=uann[1:NX+1,1:NY+1,1:NZ+1]-u[1:NX+1,1:NY+1,1:NZ+1]
  en=np.max(np.max(np.abs(error)))
  print('  cycle: ',it,', L_inf(res.)= ',rtol,',L_inf(true error): ',en)

print('Elapsed time: ',time.time()-tb,' seconds')

u,res=FMG(NX,NY,NZ,nlevels,f,1)

print('Elapsed time: ',time.time()-tb,' seconds')
error=uann[1:NX+1,1:NY+1,1:NZ+1]-u[1:NX+1,1:NY+1,1:NZ+1]
en=np.max(np.max(np.abs(error)))
print('L_inf(true error): ',en)
