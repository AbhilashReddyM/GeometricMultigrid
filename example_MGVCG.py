"""
Example showing multigrid as a preconditioner. The LinearOperator from
scipy.sparse is used to define the coefficient matrix and the preconditioner 
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator,bicgstab,cg,gmres
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from mgd2d import MGVP

def solve_sparse(solver,A, b,tol=1e-10,maxiter=500,M=None):
   num_iters = 0
   def callback(xk):
      nonlocal num_iters
      num_iters+=1
      #print(' iter:',num_iters,'   , residual :',np.max(np.abs(b-A*xk)))
   x,status=solver(A, b,tol=tol,maxiter=maxiter,callback=callback,M=M)
   return x,status,num_iters

def Laplace(nx,ny):
  '''
  Action of the Laplace matrix on a vector v
  '''
  def mv(v):
    u =np.zeros([nx+2,ny+2])
  
    u[1:nx+1,1:ny+1]=v.reshape([nx,ny])
    dx=1.0/nx; dy=1.0/ny
    Ax=1.0/dx**2; Ay=1.0/dy**2
  
    #BCs. Needs to be generalized!
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

    ut = (Ax*(u[2:nx+2,1:ny+1]+u[0:nx,1:ny+1])
        + Ay*(u[1:nx+1,2:ny+2]+u[1:nx+1,0:ny])
        - 2.0*(Ax+Ay)*u[1:nx+1,1:ny+1])
    return ut.reshape(v.shape)
  A = LinearOperator((nx*ny,nx*ny), matvec=mv)
  return A

def Uann(x,y,n):#analytical solution
   return (x**3-x)*(y**3-y)

def source(x,y,n):#RHS corresponding to above
  return 6*x*y*(x**2+ y**2 - 2)

#input
nlevels    = 8            #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 1*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 1*2**(nlevels-1) #
tol        = 1e-10      
maxiter    = 500

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
f[1:NX+1,1:NY+1]=source(XX,YY,1)

print('Multigrid preconditioned Krylov:')
print('Problem Details:')
print('NX:',NX,', NY:',NY,', tol:',tol,'MG levels: ',nlevels)

NN=NX*NY

# Get the coefficient matrix
A = Laplace(NX,NY)
b=f[1:NX+1,1:NY+1].ravel()

#get the multigrid preconditioner
M=MGVP(NX,NY,nlevels)

#start solving
tb=time.time()

u,info,iters=solve_sparse(bicgstab,A,b,tol=tol,maxiter=maxiter)
print('Without preconditioning. status:',info,', Iters: ',iters)
print('  Elapsed time: ',time.time()-tb,' seconds')

error=uann[1:NX+1,1:NY+1]-u.reshape([NX,NY])
print(' True Error :',np.max(np.max(np.abs(error))))

tb=time.time()
u,info,iters=solve_sparse(bicgstab,A,b,tol=tol,maxiter=maxiter,M=M)
print('With preconditioning. status:',info,', Iters: ',iters)
print('  Elapsed time: ',time.time()-tb,' seconds')

error=uann[1:NX+1,1:NY+1]-u.reshape([NX,NY])
print(' True Error :',np.max(np.max(np.abs(error))))

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(XX, YY, u.reshape([NX,NY]),cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.xlim([0,1])
plt.show()
