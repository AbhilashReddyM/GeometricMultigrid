"""
Example showing multigrid as a preconditioner. The LinearOperator from
scipy.sparse is used to define the coefficient matrix and the preconditioner 
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from mgd2d import MGVP

def Laplace(nx,ny):
  '''
  Action of the Laplace matrix on a vector v
  '''
  def mv(v):
    u =np.zeros([nx+2,ny+2])
    ut=np.zeros([nx,ny])
  
    u[1:nx+1,1:ny+1]=v.reshape([nx,ny])
  
    dx=1.0/nx
    dy=1.0/ny
  
    Ax=1.0/dx**2
    Ay=1.0/dy**2
  
    #BCs. Needs to be generalized!
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]
  
    for i in range(1,nx+1):
      for j in range(1,ny+1):
        ut[i-1,j-1]=(Ax*(u[i+1,j]+u[i-1,j])+Ay*(u[i,j+1]+u[i,j-1]) - 2.0*(Ax+Ay)*u[i,j])
    return ut.reshape(v.shape)
  return mv

#analytical solution
def Uann(x,y,n):
  return np.sin(2*n*np.pi*x)*np.sin(2*n*np.pi*y)

#RHS corresponding to above
def source(x,y,n):
  return -8 * (np.pi)**2 * n**2 * np.sin(2*n*np.pi*x) * np.sin(2*n*np.pi*y)

#input
max_cycles = 1           #maximum numbera of V cycles
nlevels    = 7            #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 2*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 2*2**(nlevels-1) #
tol        = 1e-7      

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

uann[1:NX+1,1:NY+1]=Uann(XX,YY,1)
f[1:NX+1,1:NY+1]=source(XX,YY,1)

print('Multigrid preconditioned GMRES:')
print('Problem Details:')
print('NX:',NX,', NY:',NY,', tol:',tol,'MG levels: ',nlevels)

NN=NX*NY

#start solving
tb=time.time()

# Get the coefficient matrix
mv=Laplace(NX,NY)
A = LinearOperator((NN,NN), matvec=mv)
b=f[1:NX+1,1:NY+1].ravel()

#get the multigrid preconditioner
M=MGVP(NX,NY,nlevels)

u,info=gmres(A,b,tol=tol,maxiter=10,M=M)


print(info)
print('Solve time: ',time.time()-tb,' seconds')

error=uann[1:NX+1,1:NY+1]-u.reshape([NX,NY])
print('error :',np.max(np.max(np.abs(error))))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, u.reshape([NX,NY]),cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.xlim([0,1])
plt.show()
