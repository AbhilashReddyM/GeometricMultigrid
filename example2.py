"""
This is an example showing how to call the mgd2d solver. 
A 4th order accurate solution is obtained with the 5pt stencil,
by using deferred correction.

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
max_cycles = 50   #maximum number of V cycles
nlevels    = 3    #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 

# Number of points is based on the number of multigrid levels as
# N=A*2**num_levels where A is an integer >=2. Smaller A is better
# This is a cell centered discretization
NX         = 4*2**nlevels 
NY         = 4*2**nlevels 

tol        =1e-4      #tolerance for the norm of the residual. set this equal to discretization error
#end input

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
u   =np.zeros([NX+2,NY+2])#approximation
f   =np.zeros([NX+2,NY+2])#RHS

#for deferred correction
uxx   = np.zeros_like(u)
uyy   = np.zeros_like(u)
uxxxx = np.zeros_like(u)
uyyyy = np.zeros_like(u)



#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

n=2 # number of waves in the solution
xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY,n)
f[1:NX+1,1:NY+1]=source(XX,YY,n)

print('mgd2d.py solver: Two Dimensional geometric multigrid solver')
print('NX:',NX,', NY:',NY,', tol:',tol,'levels: ',nlevels)
#start solving
tb=time.time()

for it in range(1,max_cycles+1):
  u,res=V_cycle(NX,NY,nlevels,u,f)
  rtol=np.linalg.norm(res)
  print('    cycle: ',it,', L2(res.)= ',rtol)
  if(rtol<tol):
    break

error=np.linalg.norm(uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1],2)/np.sqrt(NX*NY)
print(' 2nd Order::L_inf (true error): ',np.max(np.max(error)))
print(' Elapsed time: ',time.time()-tb,' seconds')

print('Improving approximation using deferred correction')

#deferred correction
#refer Leveque, p63

DX=1.0/NX
DY=1.0/NY

Ax=1.0/DX**2
Ay=1.0/DY**2



for i in range(1,NX+1):
  for j in range(1,NY+1):
    uxx[i,j]=(u[i+1,j]+u[i-1,j] - 2*u[i,j])/DX**2
    uyy[i,j]=(u[i,j+1]+u[i,j-1] - 2*u[i,j])/DY**2

# we should be using one-sided difference formulae for values 
# near the boundary. For simplicity I am just applying the  
# condition known from the analytical form for these terms.

uxx[ 0,:] = -uxx[ 1,:]
uxx[-1,:] = -uxx[-2,:]
uxx[:, 0] = -uxx[:, 1]
uxx[:,-1] = -uxx[:,-2]

uyy[ 0,:] = -uyy[ 1,:]
uyy[-1,:] = -uyy[-2,:]
uyy[:, 0] = -uyy[:, 1]
uyy[:,-1] = -uyy[:,-2]

for i in range(1,NX+1):
  for j in range(1,NY+1):
    uxxxx[i,j]=(uxx[i+1,j]+uxx[i-1,j] - 2*uxx[i,j])/DX**2
    uyyyy[i,j]=(uyy[i,j+1]+uyy[i,j-1] - 2*uyy[i,j])/DY**2

#adjust the RHS to cancel the leading order terms
for i in range(1,NX+1):
  for j in range(1,NY+1):
    f[i,j]+= + 1.0/12*DX**2*(uxxxx[i,j]+uyyyy[i,j])

#solve once again with the new RHS
for it in range(1,max_cycles+1):
  u,res=V_cycle(NX,NY,nlevels,u,f)
  rtol=np.linalg.norm(res)
  print('    cycle: ',it,', L2(res.)= ',rtol)
  if(rtol<tol):
    break

tf=time.time()
error=np.linalg.norm(uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1],2)/np.sqrt(NX*NY)
print(' 4nd Order::L_inf (true error): ',np.max(np.max(error)))
print('Elapsed time: ',tf-tb,' seconds')


