"""
This is an example showing how to use the mgd2d solver. 
A 4th order accurate solution is obtained with the 5pt stencil,
by using deferred correction.

"""
import numpy as np
import time
from mgd2d import FMG,V_cycle
#analytical solution
def Uann(x,y,n):
  return np.sin(2*n*np.pi*x)*np.sin(2*n*np.pi*y)

#RHS corresponding to above
def source(x,y,n):
  return -8 * (np.pi)**2 * n**2 * np.sin(2*n*np.pi*x) * np.sin(2*n*np.pi*y)

#input

#input
#FMG is a direct solver. tolerance and iterations are not used 

#nv          = 1 # nv : Number of V-cycles within FMG. nv=1 will give solution a to within discretization error.
                 # Increase this to get a higher accuracy solution (upto roundoff limit) 
                 # of the discrete problem.(residual on the fine grid =round off limit)
                 # Here I am using nv=2 for the first solve and nv=6 for the second solve.

nlevels    = 7    #total number of grid levels. 1 means no multigrid, 2 means one coarse grid. etc 

# Number of points is based on the number of multigrid levels as
# N=A*2**(num_levels-1) where A is an integer >=4. Smaller A is better
# This is a cell centered discretization
NX         = 4*2**(nlevels-1) 
NY         = 4*2**(nlevels-1) 

#the grid has one layer of ghost cells to help apply the boundary conditions
uann=np.zeros([NX+2,NY+2])#analytical solution
u   =np.zeros([NX+2,NY+2])#approximation
f   =np.zeros([NX+2,NY+2])#RHS

#for deferred correction
uxx   = np.zeros_like(u)
corr  = np.zeros_like(u)

#calcualte the RHS and exact solution
DX=1.0/NX
DY=1.0/NY

n=1 # number of waves in the solution
xc=np.linspace(0.5*DX,1-0.5*DX,NX)
yc=np.linspace(0.5*DY,1-0.5*DY,NY)
XX,YY=np.meshgrid(xc,yc,indexing='ij')

uann[1:NX+1,1:NY+1]=Uann(XX,YY,n)
f[1:NX+1,1:NY+1]=source(XX,YY,n)

print('mgd2d.py : Two Dimensional geometric multigrid solver')
print('NX:',NX,', NY:',NY,', levels: ',nlevels)
#start solving
tb=time.time()

u,res=FMG(NX,NY,nlevels,f,2)

error=np.abs(uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1])
print(' 2nd Order::L_inf (true error): ',np.max(np.max(error)))
print(' Elapsed time: ',time.time()-tb,' seconds')

print('Improving approximation using deferred correction')

#deferred correction
#refer Leveque, p63
Ax=1.0/DX**2
Ay=1.0/DY**2

for i in range(1,NX+1):
  for j in range(1,NY+1):
    uxx[i,j]=(u[i+1,j]+u[i-1,j] - 2*u[i,j])/DX**2

# we should be using one-sided difference formulae for values 
# near the boundary. For simplicity I am just applying the  
# condition known from the analytical form for these terms.

uxx[ 0,:] = -uxx[ 1,:]
uxx[-1,:] = -uxx[-2,:]
uxx[:, 0] = -uxx[:, 1]
uxx[:,-1] = -uxx[:,-2]

f[ 0,:] = -f[ 1,:]
f[-1,:] = -f[-2,:]
f[:, 0] = -f[:, 1]
f[:,-1] = -f[:,-2]

#correction term
#  del2(f)-2*uxxyy
for i in range(1,NX+1):
  for j in range(1,NY+1):
    corr[i,j]=(Ax*(f[i+1,j]+f[i-1,j])+Ay*(f[i,j+1]+f[i,j-1])-2.0*(Ax+Ay)*f[i,j])-2*(uxx[i,j+1]+uxx[i,j-1] - 2*uxx[i,j])/DY**2

#adjust the RHS to cancel the leading order terms
for i in range(1,NX+1):
  for j in range(1,NY+1):
    f[i,j]+= 1.0/12*DX**2*(corr[i,j])

##solve once again with the new RHS
u,res=FMG(NX,NY,nlevels,f,5)

tf=time.time()
error=np.abs(uann[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1])
print(' 4nd Order::L_inf (true error): ',np.max(np.max(error)))
print('Elapsed time: ',tf-tb,' seconds')


