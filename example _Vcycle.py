"""
This is an example showing how to call the mgd2d solver.
"""
import numpy as np
import time
from mgd2d import V_cycle,FMG

#analytical solution
def Uann(x,y,n):
  return np.sin(2*n*np.pi*x)*np.sin(2*n*np.pi*y)

#RHS corresponding to above
def source(x,y,n):
  return -8 * (np.pi)**2 * n**2 * np.sin(2*n*np.pi*x) * np.sin(2*n*np.pi*y)

#input
max_cycles = 50           #maximum number of V cycles
nlevels    = 7            #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 4*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 4*2**(nlevels-1) #
tol        = 1e-9      

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
u   =np.zeros([NX+2,NY+2])#approximation
f   =np.zeros([NX+2,NY+2])#RHS

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

n=1.0 # number of waves in the solution
xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY,1)
f[1:NX+1,1:NY+1]=source(XX,YY,1)

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



