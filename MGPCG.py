"""
2017 A. R. Malipeddi
Multigrid preconditioned Conjugate gradient method
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,bicgstab,cg,gmres
import time

from mgd2d import MGVP

def solve_sparse(solver,A, b,tol=1e-10,maxiter=500,M=None):
      num_iters = 0
      def callback(xk):
         nonlocal num_iters
         num_iters+=1
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

#input
nlevels    = 8                #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 
NX         = 1*2**(nlevels-1) #Nx and Ny are given as function of grid levels
NY         = 1*2**(nlevels-1) 
tol        = 1e-10      
maxiter    = 500

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

print('Multigrid preconditioned krylov:')
print('Problem Details:')
print('NX:',NX,', NY:',NY,', tol:',tol,'MG levels: ',nlevels)


#Laplace Operator
A = Laplace(NX,NY)

#get a random solution and the rhs for this solution by applying the matrix 
uex=np.random.rand(NX*NY,1)
b=A*uex

#Multigrid Preconditioner
M=MGVP(NX,NY,nlevels)

#start solving
tb=time.time()

#start solving
u,info,iters=solve_sparse(cg,A,b,tol=tol,maxiter=maxiter)
print('Without preconditioning. status:',info,', Iters: ',iters)
print('  Elapsed time: ',time.time()-tb,' seconds')
error=uex.reshape([NX,NY])-u.reshape([NX,NY])
print('  Error :',np.max(np.max(np.abs(error))))

#start solving
tb=time.time()
u,info,iters=solve_sparse(cg,A,b,tol=tol,maxiter=maxiter,M=M)
print('With preconditioning. status:',info,', Iters: ',iters)
print('  Elapsed time: ',time.time()-tb,' seconds')
error=uex.reshape([NX,NY])-u.reshape([NX,NY])
print(' Error :',np.max(np.max(np.abs(error))))

