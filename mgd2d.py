"""
2017 A. R. Malipeddi
A simple 2D geometric multigrid solver for the homogeneous Dirichlet Poisson problem on Cartesian grids and unit square. Cell centered 5-point finite difference operator.
"""

import numpy as np

def GSrelax(nx,ny,u,f,iters=1):
  '''
  Gauss Seidel smoothing
  '''
  
  dx=1.0/nx;
  dy=1.0/ny

  Ax=1.0/dx**2
  Ay=1.0/dy**2
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
  for i in range(1,nx+1):
    for j in range(1,ny+1):
      res[i,j]=f[i,j] - ((Ax*(u[i+1,j]+u[i-1,j])+Ay*(u[i,j+1]+u[i,j-1]) - 2.0*(Ax+Ay)*u[i,j]))
  return u,res

def restrict(nx,ny,v):
  '''
  restrict 'v' to the coarser grid
  '''
  v_c=np.zeros([nx+2,ny+2])
  
  for i in range(1,nx+1):
    for j in range(1,ny+1):
      v_c[i,j]=0.25*(v[2*i-1,2*j-1]+v[2*i,2*j-1]+v[2*i-1,2*j]+v[2*i,2*j])

  return v_c

def prolong(nx,ny,v):
  '''
  interpolate 'v' to the fine grid
  '''
  v_f=np.zeros([2*nx+2,2*ny+2])

  for i in range(1,nx+1):
    for j in range(1,ny+1):
      v_f[2*i-1,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j-1])+0.0625*v[i-1,j-1]
      v_f[2*i  ,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j-1])+0.0625*v[i+1,j-1]
      v_f[2*i-1,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j+1])+0.0625*v[i-1,j+1]
      v_f[2*i  ,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j+1])+0.0625*v[i+1,j+1]

  return v_f

def V_cycle(nx,ny,num_levels,u,f,level=1):
  '''
  V cycle
  '''
  if(level==num_levels):#bottom solve
    u,res=GSrelax(nx,ny,u,f,iters=50)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=GSrelax(nx,ny,u,f,iters=1)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(nx//2,ny//2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(nx//2,ny//2,num_levels,e_c,res_c,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u+=prolong(nx//2,ny//2,e_c)
  
  #Step 5: Relax Au=f on this grid
  u,res=GSrelax(nx,ny,u,f,iters=1)
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


