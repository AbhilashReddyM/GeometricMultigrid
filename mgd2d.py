"""
2017 A. R. Malipeddi
A simple 2D geometric multigrid solver for the homogeneous Dirichlet Poisson problem on Cartesian grids and unit square. Cell centered 5-point finite difference operator.
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator

def Jacrelax(nx,ny,u,f,iters=1):
  '''
  Gauss Seidel smoothing
  '''

  dx=1.0/nx; dy=1.0/ny
  Ax=1.0/dx**2; Ay=1.0/dy**2
  Ap=1.0/(2.0*(Ax+Ay))

  #BCs. Needs to be generalized!
  u[ 0,:] = -u[ 1,:]
  u[-1,:] = -u[-2,:]
  u[:, 0] = -u[:, 1]
  u[:,-1] = -u[:,-2]

  for it in range(iters):
    u[1:nx+1,1:ny+1] = 0.8*Ap*(Ax*(u[2:nx+2,1:ny+1] + u[0:nx,1:ny+1])
                             + Ay*(u[1:nx+1,2:ny+2] + u[1:nx+1,0:ny])
                             - f[1:nx+1,1:ny+1])+0.2*u[1:nx+1,1:ny+1]
    #Dirichlet BC
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

  res=np.zeros([nx+2,ny+2])

  res[1:nx+1,1:ny+1]=f[1:nx+1,1:ny+1]-(( Ax*(u[2:nx+2,1:ny+1]+u[0:nx,1:ny+1])
                                       + Ay*(u[1:nx+1,2:ny+2]+u[1:nx+1,0:ny])
                                       - 2.0*(Ax+Ay)*u[1:nx+1,1:ny+1]))
  return u,res

def SGSrelax(nx,ny,u,f,iters=1):
  '''
  Gauss Seidel smoothing
  '''
  dx=1.0/nx; dy=1.0/ny
  Ax=1.0/dx**2; Ay=1.0/dy**2
  Ap=1.0/(2.0*(Ax+Ay))

  #BCs. Needs to be generalized!
  u[ 0,:] = -u[ 1,:]
  u[-1,:] = -u[-2,:]
  u[:, 0] = -u[:, 1]
  u[:,-1] = -u[:,-2]

  for it in range(iters):
    for i in range(1,nx+1):
     for j in range(1,ny+1):
         u[i,j]= Ap*( Ax*(u[i+1,j]+u[i-1,j])
                     +Ay*(u[i,j+1]+u[i,j-1]) - f[i,j])
    #BCs. Needs to be generalized!
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

    for i in reversed(range(1,nx+1)):
     for j in reversed(range(1,ny+1)):
         u[i,j]= 0.66*(Ap*( Ax*(u[i+1,j]+u[i-1,j])
                     +Ay*(u[i,j+1]+u[i,j-1]) - f[i,j])) + 0.34*u[i,j]
    #BCs. Needs to be generalized!
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

  res=np.zeros([nx+2,ny+2])
  for i in range(1,nx+1):
    for j in range(1,ny+1):
      res[i,j]=f[i,j] - ((Ax*(u[i+1,j]+u[i-1,j])+Ay*(u[i,j+1]+u[i,j-1]) - 2.0*(Ax+Ay)*u[i,j]))
  return u,res

def GSrelax(nx,ny,u,f,iters=1):
  '''
  Gauss Seidel smoothing
  '''
  dx=1.0/nx; dy=1.0/ny

  Ax=1.0/dx**2;  Ay=1.0/dy**2
  Ap=1.0/(2.0*(Ax+Ay))

  #BCs. Needs to be generalized!
  u[ 0,:] = -u[ 1,:]
  u[-1,:] = -u[-2,:]
  u[:, 0] = -u[:, 1]
  u[:,-1] = -u[:,-2]

  for it in range(iters):
    for c in [0,1]:#Red Black ordering
     for i in range(1,nx+1):
      start = 1 + (i%2) if c == 0 else 2 - (i%2)
      for j in range(start,ny+1,2):
         u[i,j]= Ap*( Ax*(u[i+1,j]+u[i-1,j])
                     +Ay*(u[i,j+1]+u[i,j-1]) - f[i,j])
    #BCs. Needs to be generalized!
    u[ 0,:] = -u[ 1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

  res=np.zeros([nx+2,ny+2])

  res[1:nx+1,1:ny+1]=f[1:nx+1,1:ny+1]-(( Ax*(u[2:nx+2,1:ny+1]+u[0:nx,1:ny+1])
                                       + Ay*(u[1:nx+1,2:ny+2]+u[1:nx+1,0:ny])
                                       - 2.0*(Ax+Ay)*u[1:nx+1,1:ny+1]))
  return u,res

def restrict(nx,ny,v):
  '''
  restrict 'v' to the coarser grid
  '''
  v_c=np.zeros([nx+2,ny+2])

#  #vecrorized form of 
#  for i in range(1,nx+1):
#    for j in range(1,ny+1):
#      v_c[i,j]=0.25*(v[2*i-1,2*j-1]+v[2*i,2*j-1]+v[2*i-1,2*j]+v[2*i,2*j])
  
  v_c[1:nx+1,1:ny+1]=0.25*(v[1:2*nx:2,1:2*ny:2]+v[1:2*nx:2,2:2*ny+1:2]+v[2:2*nx+1:2,1:2*ny:2]+v[2:2*nx+1:2,2:2*ny+1:2])

  return v_c

def prolong(nx,ny,v):
  '''
  interpolate 'v' to the fine grid
  '''
  v_f=np.zeros([2*nx+2,2*ny+2])

#  #vecrorized form of 
#  for i in range(1,nx+1):
#    for j in range(1,ny+1):
#      v_f[2*i-1,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j-1])+0.0625*v[i-1,j-1]
#      v_f[2*i  ,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j-1])+0.0625*v[i+1,j-1]
#      v_f[2*i-1,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j+1])+0.0625*v[i-1,j+1]
#      v_f[2*i  ,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j+1])+0.0625*v[i+1,j+1]

  v_f[1:2*nx:2  ,1:2*ny:2  ] = 0.5625*v[1:nx+1,1:ny+1]+0.1875*(v[0:nx  ,1:ny+1]+v[1:nx+1,0:ny]  )+0.0625*v[0:nx  ,0:ny  ]
  v_f[2:2*nx+1:2,1:2*ny:2  ] = 0.5625*v[1:nx+1,1:ny+1]+0.1875*(v[2:nx+2,1:ny+1]+v[1:nx+1,0:ny]  )+0.0625*v[2:nx+2,0:ny  ]
  v_f[1:2*nx:2  ,2:2*ny+1:2] = 0.5625*v[1:nx+1,1:ny+1]+0.1875*(v[0:nx  ,1:ny+1]+v[1:nx+1,2:ny+2])+0.0625*v[0:nx  ,2:ny+2]
  v_f[2:2*nx+1:2,2:2*ny+1:2] = 0.5625*v[1:nx+1,1:ny+1]+0.1875*(v[2:nx+2,1:ny+1]+v[1:nx+1,2:ny+2])+0.0625*v[2:nx+2,2:ny+2]

  return v_f

def V_cycle(nx,ny,num_levels,u,f,level=1):
  '''
  V cycle
  '''
  if(level==num_levels):#bottom solve
    u,res=Jacrelax(nx,ny,u,f,iters=50)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=Jacrelax(nx,ny,u,f,iters=1)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(nx//2,ny//2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(nx//2,ny//2,num_levels,e_c,res_c,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u+=prolong(nx//2,ny//2,e_c)
  
  #Step 5: Relax Au=f on this grid
  u,res=Jacrelax(nx,ny,u,f,iters=1)
  return u,res

def FMG(nx,ny,num_levels,f,nv=1,level=1):

  if(level==num_levels):#bottom solve
    u=np.zeros([nx+2,ny+2])  
    u,res=GSrelax(nx,ny,u,f,iters=50)
    return u,res

  #Step 1: Restrict the rhs to a coarse grid
  f_c=restrict(nx//2,ny//2,f)

  #Step 2: Solve the coarse grid problem using FMG
  u_c,_=FMG(nx//2,ny//2,num_levels,f_c,nv,level+1)

  #Step 3: Interpolate u_c to the fine grid
  u=prolong(nx//2,ny//2,u_c)

  #step 4: Execute 'nv' V-cycles
  for _ in range(nv):
    u,res=V_cycle(nx,ny,num_levels-level,u,f)
  return u,res

def MGVP(nx,ny,num_levels):
  '''
  Multigrid Preconditioner. Returns a (scipy.sparse) LinearOperator that can
  be passed to Krylov solvers as a preconditioner. The matrix is not 
  explicitly needed.  All that is needed is a matrix vector product 
  In any stationary iterative method, the preconditioner-vector product
  can be obtained by setting the RHS to the vector and initial guess to 
  zero and performing one iteration. (Richardson Method)  
  '''
  def pc_fn(v):

    u =np.zeros([nx+2,ny+2])
    f =np.zeros([nx+2,ny+2])
    f[1:nx+1,1:ny+1] =v.reshape([nx,ny])

    #perform one V cycle
    u,res=V_cycle(nx,ny,num_levels,u,f)

    return u[1:nx+1,1:ny+1].reshape(v.shape)

  M=LinearOperator((nx*ny,nx*ny), matvec=pc_fn)

  return M

