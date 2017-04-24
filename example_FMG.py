"""
This is an example showing how to call the mgd2d FMG solver.
"""
import numpy as np
import time
from mgd2d import FMG

def Uann(x,y,n):#analytical solution
   return (x**3-x)*(y**3-y)

def source(x,y,n):#RHS corresponding to above
  return 6*x*y*(x**2+ y**2 - 2)

#input
#FMG is a direct solver. tolerance and iterations are not used 
nv          = 1     # nv : Number of V-cycles within FMG. nv=1 will give solution a to within discretization error.
                    # Increase this (to ~5) to get exact solution (upto roundoff limit) of the discrete problem
                    # For instance, when deferred correction is desired.
nlevels     = 8     # total number of grid levels. Also determines the grid size 
NX          = 1*2**(nlevels-1) 
NY          = 1*2**(nlevels-1) 

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
f   =np.zeros([NX+2,NY+2])#RHS

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY,1)
f[1:NX+1,1:NY+1]   =source(XX,YY,1)

print('mgd2d.py FMG solver:')
print('NX:',NX,', NY:',NY,', levels: ',nlevels)

# solving
tb=time.time()
u,res=FMG(NX,NY,nlevels,f,nv) 
tf=time.time()

rtol=np.max(np.max(np.abs(res)))
print(' FMG L_inf(res.)= ',rtol)

print('Elapsed time: ',tf-tb,' seconds')
error=uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1]
print('L_inf (true error): ',np.max(np.max(np.abs(error))))
