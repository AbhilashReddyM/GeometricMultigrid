"""
This is an example showing how to call the mgd2d solver.
"""
import numpy as np
import time
from mgd2d import V_cycle
#analytical solution
def Uann(x,y,n):
  return np.sin(2*n*np.pi*x)*np.sin(2*n*np.pi*y)

#RHS corresponding to above
def source(x,y,n):
  return -8 * (np.pi)**2 * n**2 * np.sin(2*n*np.pi*x) * np.sin(2*n*np.pi*y)

#input
max_cycles = 50           #maximum number of V cycles
nlevels    = 5            #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 

# Number of points is based on the number of multigrid levels as
# N=A*2**num_levels where A is an integer >=2. Smaller A is better
# This is a cell centered discretization
NX         = 4*2**nlevels 
NY         = 4*2**nlevels 

tol        =1e-5      #tolerance for the norm of the residual. set this equal to discretization error
#end input

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
u   =np.zeros([NX+2,NY+2])#approximation
f   =np.zeros([NX+2,NY+2])#RHS


#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

n=1 # number of waves in the solution
xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY,n)
f[1:NX+1,1:NY+1]=source(XX,YY,n)

print('mgd2d.py solver:')
print('NX:',NX,', NY:',NY,', tol:',tol,'levels: ',nlevels)
#start solving
tb=time.time()

for it in range(1,max_cycles+1):
  u,res=V_cycle(NX,NY,nlevels,u,f)
  rtol=np.linalg.norm(res)
  print('cycle: ',it,', L2(res.)= ',rtol)
  if(rtol<tol):
    break

tf=time.time()
print('Solve time: ',tf-tb,' seconds')
error=np.abs(uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1])
print('L_inf (true error): ',np.max(np.max(error)))