"""
2017 A. R. Malipeddi
A simple 2D geometric multigrid solver for the homogeneous Dirichlet Poisson problem on Cartesian grids and unit square. Cell centered 5-point finite difference operator.
"""

import numpy as np

def GSrelax(nx,ny,u,f,iters=1,RF=1.0,flag=1):
  '''
  Red-Black Gauss Seidel smoothing
  flag : 1 = pre-sweep
         2 = post-sweep
  '''
  
  dx=1.0/nx
  dy=1.0/ny

  Ax=1.0/dx**2
  Ay=1.0/dy**2
  Ap=1.0/(2.0*(1.0/dx**2+1.0/dy**2))
  
  for it in range(iters):
   for c in [0,1]:
    for i in range(1,nx+1):
     start = 1 + (i%2) if c == 0 else 2 - (i%2)
     for j in range(start,ny+1,2):
        u[i,j]= RF*( Ax*(u[i+1,j]+u[i-1,j])+Ay*(u[i,j+1]+u[i,j-1]) - f[i,j])*Ap +(1.0-RF)*u[i,j]

   #homogeneous dirichlet BCs. Needs to be generalized!
   #The BCs for the residual equation will be homogeneous regardless of the BCs for the PDE. 
   u[ 0,:] = -u[ 1,:]
   u[-1,:] = -u[-2,:]
   u[:, 0] = -u[:, 1]
   u[:,-1] = -u[:,-2]

  #residual not needed for post sweep
  if(flag==2):
    return u,None

  res=np.zeros([nx+2,ny+2])

  for i in range(1,nx+1):
    for j in range(1,ny+1):
      res[i,j]=f[i,j] - ((Ax*(u[i+1,j]+u[i-1,j])+Ay*(u[i,j+1]+u[i,j-1]) - 2.0*(Ax+Ay)*u[i,j]))

  return u,res

def restrict(nx,ny,res):
  '''
  restrict the residual to the coarser grid
  '''
  res_c=np.zeros([nx+2,ny+2])
  
  for i in range(1,nx+1):
    for j in range(1,ny+1):
      res_c[i,j]=0.25*(res[2*i-1,2*j-1]+res[2*i,2*j-1]+res[2*i-1,2*j]+res[2*i,2*j])
  return res_c

def prolong(u,nx,ny,e_c):
  '''
  interpolate correction and add to the fine grid approximation
  '''
  for i in range(1,nx+1):
    for j in range(1,ny+1):
      u[2*i-1,2*j-1] += 0.5*e_c[i,j]+0.25*(e_c[i-1,j]+e_c[i,j-1])
      u[2*i  ,2*j-1] += 0.5*e_c[i,j]+0.25*(e_c[i+1,j]+e_c[i,j-1])
      u[2*i-1,2*j  ] += 0.5*e_c[i,j]+0.25*(e_c[i-1,j]+e_c[i,j+1])
      u[2*i  ,2*j  ] += 0.5*e_c[i,j]+0.25*(e_c[i+1,j]+e_c[i,j+1])

  return u

def V_cycle(nx,ny,num_levels,u,f,level=1):
  '''
  V cycle
  '''

  if(level==num_levels):#lowest level reached
    u,res=GSrelax(nx,ny,u,f,50,RF=1.2,flag=2)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=GSrelax(nx,ny,u,f,iters=2)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(nx//2,ny//2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(nx//2,ny//2,num_levels,e_c,res_c,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u=prolong(u,nx//2,ny//2,e_c)
  
  #Step 5: Relax Au=f on this grid
  if(level==1):
    u,res=GSrelax(nx,ny,u,f,iters=1)
  else:
    u,res=GSrelax(nx,ny,u,f,iters=1,flag=2)
  return u,res
