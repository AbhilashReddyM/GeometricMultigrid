"""
2017 (c) A. R. Malipeddi
3D geometric multigrid code for poissons equation in a cube.
 - Finite difference method
 - 7pt operator
 - trilinear interpolation
 - Jacobi smoothing

"""

import numpy as np

def Jacrelax(level,nx,ny,nz,u,f,iters=1,pre=False):
  '''
  Jacobi method smoothing
  '''
  dx=1.0/nx;    dy=1.0/ny;    dz=1.0/nz
  Ax=1.0/dx**2; Ay=1.0/dy**2; Az=1.0/dz**2
  Ap=1.0/(2.0*(1.0/dx**2+1.0/dy**2+1.0/dz**2))

  #Dirichlet BC
  u[ 0,:,:] = -u[ 1,:,:]
  u[-1,:,:] = -u[-2,:,:]
  u[: ,0,:] = -u[:, 1,:]
  u[:,-1,:] = -u[:,-2,:]
  u[:,:, 0] = -u[:,:, 1]
  u[:,:,-1] = -u[:,:,-2]

  #if it is a pre-sweep not on the finest grid u is fully zero and only the f term contributes
  # in the first iteration. This avoids some calculation. Additional iterations are as usual. 
  if(pre and level>1):
    u[1:nx+1,1:ny+1,1:nz+1] = -Ap*f[1:nx+1,1:ny+1,1:nz+1]
    #Dirichlet BC
    u[ 0,:,:] = -u[ 1,:,:]
    u[-1,:,:] = -u[-2,:,:]
    u[: ,0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]
    iters=iters-1
  
  for it in range(iters):
    u[1:nx+1,1:ny+1,1:nz+1] = Ap*(Ax*(u[2:nx+2,1:ny+1,1:nz+1] + u[0:nx,1:ny+1,1:nz+1])
                                + Ay*(u[1:nx+1,2:ny+2,1:nz+1] + u[1:nx+1,0:ny,1:nz+1])
                                + Az*(u[1:nx+1,1:ny+1,2:nz+2] + u[1:nx+1,1:ny+1,0:nz])
                                - f[1:nx+1,1:ny+1,1:nz+1])
    #Dirichlet BC
    u[ 0,:,:] = -u[ 1,:,:]
    u[-1,:,:] = -u[-2,:,:]
    u[: ,0,:] = -u[:, 1,:]
    u[:,-1,:] = -u[:,-2,:]
    u[:,:, 0] = -u[:,:, 1]
    u[:,:,-1] = -u[:,:,-2]

#if it is a post sweep then dont need residual, so we can return here. We will need residual if it is the post sweep on the fine grid though (to return along with the solution)
#  if(not pre):
#    return u,None

  res=np.zeros([nx+2,ny+2,nz+2])
  res[1:nx+1,1:ny+1,1:nz+1]=f[1:nx+1,1:ny+1,1:nz+1]-(Ax*(u[2:nx+2,1:ny+1,1:nz+1] + u[0:nx,1:ny+1,1:nz+1])
                                                   + Ay*(u[1:nx+1,2:ny+2,1:nz+1] + u[1:nx+1,0:ny,1:nz+1])
                                                   + Az*(u[1:nx+1,1:ny+1,2:nz+2] + u[1:nx+1,1:ny+1,0:nz])
                                                   - 2.0*(Ax+Ay+Az)*u[1:nx+1,1:ny+1,1:nx+1])
  return u,res


def restrict(nx,ny,nz,v):
  '''
  restrict 'v' to the coarser grid
  '''
  v_c=np.zeros([nx+2,ny+2,nz+2])

  v_c[1:nx+1,1:ny+1,1:nz+1]=0.125*(v[1:2*nx:2,1:2*ny:2,1:2*nz:2  ]+v[1:2*nx:2,2:2*ny+1:2,1:2*nz:2  ]+v[2:2*nx+1:2,1:2*ny:2,1:2*nz:2  ]+v[2:2*nx+1:2,2:2*ny+1:2,1:2*nz:2  ]
                                 + v[1:2*nx:2,1:2*ny:2,2:2*nz+1:2]+v[1:2*nx:2,2:2*ny+1:2,2:2*nz+1:2]+v[2:2*nx+1:2,1:2*ny:2,2:2*nz+1:2]+v[2:2*nx+1:2,2:2*ny+1:2,2:2*nz+1:2])
  return v_c


def prolong(nx,ny,nz,v):
  '''
  interpolate correction to the fine grid
  '''
  v_f=np.zeros([2*nx+2,2*ny+2,2*nz+2])

  a=27.0/64
  b= 9.0/64
  c= 3.0/64
  d= 1.0/64

  v_f[1:2*nx:2  ,1:2*ny:2  ,1:2*nz:2  ] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[0:nx  ,1:ny+1,1:nz+1] + v[1:nx+1,0:ny  ,1:nz+1] + v[1:nx+1,1:ny+1,0:nz  ]) + c*(v[0:nx  ,0:ny  ,1:nz+1] + v[0:nx  ,1:ny+1,0:nz  ] + v[1:nx+1,0:ny  ,0:nz  ]) + d*v[0:nx  ,0:ny  ,0:nz  ]
  v_f[2:2*nx+1:2,1:2*ny:2  ,1:2*nz:2  ] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[2:nx+2,1:ny+1,1:nz+1] + v[1:nx+1,0:ny  ,1:nz+1] + v[1:nx+1,1:ny+1,0:nz  ]) + c*(v[2:nx+2,0:ny  ,1:nz+1] + v[2:nx+2,1:ny+1,0:nz  ] + v[1:nx+1,0:ny  ,0:nz  ]) + d*v[2:nx+2,0:ny  ,0:nz  ]
  v_f[1:2*nx:2  ,2:2*ny+1:2,1:2*nz:2  ] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[0:nx  ,1:ny+1,1:nz+1] + v[1:nx+1,2:ny+2,1:nz+1] + v[1:nx+1,1:ny+1,0:nz  ]) + c*(v[0:nx  ,2:ny+2,1:nz+1] + v[0:nx  ,1:ny+1,0:nz  ] + v[1:nx+1,2:ny+2,0:nz  ]) + d*v[0:nx  ,2:ny+2,0:nz  ]
  v_f[2:2*nx+1:2,2:2*ny+1:2,1:2*nz:2  ] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[2:nx+2,1:ny+1,1:nz+1] + v[1:nx+1,2:ny+2,1:nz+1] + v[1:nx+1,1:ny+1,0:nz  ]) + c*(v[2:nx+2,2:ny+2,1:nz+1] + v[2:nx+2,1:ny+1,0:nz  ] + v[1:nx+1,2:ny+2,0:nz  ]) + d*v[2:nx+2,2:ny+2,0:nz  ]
  v_f[1:2*nx:2  ,1:2*ny:2  ,2:2*nz+1:2] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[0:nx  ,1:ny+1,1:nz+1] + v[1:nx+1,0:ny  ,1:nz+1] + v[1:nx+1,1:ny+1,2:nz+2]) + c*(v[0:nx  ,0:ny  ,1:nz+1] + v[0:nx  ,1:ny+1,2:nz+2] + v[1:nx+1,0:ny  ,2:nz+2]) + d*v[0:nx  ,0:ny  ,2:nz+2]
  v_f[2:2*nx+1:2,1:2*ny:2  ,2:2*nz+1:2] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[2:nx+2,1:ny+1,1:nz+1] + v[1:nx+1,0:ny  ,1:nz+1] + v[1:nx+1,1:ny+1,2:nz+2]) + c*(v[2:nx+2,0:ny  ,1:nz+1] + v[2:nx+2,1:ny+1,2:nz+2] + v[1:nx+1,0:ny  ,2:nz+2]) + d*v[2:nx+2,0:ny  ,2:nz+2]
  v_f[1:2*nx:2  ,2:2*ny+1:2,2:2*nz+1:2] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[0:nx  ,1:ny+1,1:nz+1] + v[1:nx+1,2:ny+2,1:nz+1] + v[1:nx+1,1:ny+1,2:nz+2]) + c*(v[0:nx  ,2:ny+2,1:nz+1] + v[0:nx  ,1:ny+1,2:nz+2] + v[1:nx+1,2:ny+2,2:nz+2]) + d*v[0:nx  ,2:ny+2,2:nz+2]
  v_f[2:2*nx+1:2,2:2*ny+1:2,2:2*nz+1:2] = a*v[1:nx+1,1:ny+1,1:nz+1] + b*(v[2:nx+2,1:ny+1,1:nz+1] + v[1:nx+1,2:ny+2,1:nz+1] + v[1:nx+1,1:ny+1,2:nz+2]) + c*(v[2:nx+2,2:ny+2,1:nz+1] + v[2:nx+2,1:ny+1,2:nz+2] + v[1:nx+1,2:ny+2,2:nz+2]) + d*v[2:nx+2,2:ny+2,2:nz+2]

  return v_f


def V_cycle(nx,ny,nz,num_levels,u,f,level=1):
  '''
  V cycle
  '''
  if(level==num_levels):#bottom solve
    u,res=Jacrelax(level,nx,ny,nz,u,f,iters=100,pre=True)
    return u,res

  #Step 1: Relax Au=f on this grid
  u,res=Jacrelax(level,nx,ny,nz,u,f,1,True)

  #Step 2: Restrict residual to coarse grid
  res_c=restrict(nx//2,ny//2,nz//2,res)

  #Step 3:Solve A e_c=res_c on the coarse grid. (Recursively)
  e_c=np.zeros_like(res_c)
  e_c,res_c=V_cycle(nx//2,ny//2,nz//2,num_levels,e_c,res_c,level+1)

  #Step 4: Interpolate(prolong) e_c to fine grid and add to u
  u+=prolong(nx//2,ny//2,nz//2,e_c)

  #Step 5: Relax Au=f on this grid
  u,res=Jacrelax(level,nx,ny,nz,u,f,1,False)

  return u,res


def FMG(nx,ny,nz,num_levels,f,nv=1,level=1):

  if(level==num_levels):#bottom solve
    u=np.zeros([nx+2,ny+2,nz+2])
    u,res=Jacrelax(level,nx,ny,nz,u,f,iters=100,pre=True)
    return u,res

  #Step 1: Restrict the rhs to a coarse grid
  f_c=restrict(nx//2,ny//2,nz//2,f)

  #Step 2: Solve the coarse grid problem using FMG
  u_c,_=FMG(nx//2,ny//2,nz//2,num_levels,f_c,nv,level+1)

  #Step 3: Interpolate u_c to the fine grid
  u=prolong(nx//2,ny//2,nz//2,u_c)

  #step 4: Execute 'nv' V-cycles
  for _ in range(nv):
    u,res=V_cycle(nx,ny,nz,num_levels-level,u,f)
  return u,res



