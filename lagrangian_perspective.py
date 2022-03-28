import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Generate a quadratic program with respect to [x1,x2,x3,x4]
N=100
dt=1/N
x1 = cp.Variable(N)
x2 = cp.Variable(N)
x3 = cp.Variable(N)
x4 = cp.Variable(N)
b1 = cp.Variable(N)
b2 = cp.Variable(N)
b3 = cp.Variable(N)
b4 = cp.Variable(N)


x1_0=cp.Parameter()
x2_0=cp.Parameter()
x3_0=cp.Parameter()
x4_0=cp.Parameter()

obje = dt*sum(5*b2+9*b4)+1000*x3[N-1]

cons = [x1[0]==x1_0]\
+[x2[0]==x2_0]\
+[x3[0]==x3_0]\
+[x4[0]==x4_0]\
+[x1[k+1]==x1[k]-b1[k]*dt for k in range(N-1)]\
+[x2[k+1]==x2[k]-b2[k]*dt for k in range(N-1)]\
+[x3[k+1]==x3[k]-b3[k]*dt for k in range(N-1)]\
+[x4[k+1]==x4[k]-b4[k]*dt for k in range(N-1)]\
+[b1[k]==-x2[k] for k in range(N)]\
+[b3[k]==-x4[k] for k in range(N)]\
+[-b2[k]-b4[k]<=0 for k in range(N)]\
+[2*b2[k]+b4[k]<=0 for k in range(N)]\
+[5*b2[k]+9*b4[k]<=1 for k in range(N)]\
+[x2[k]<=0.1 for k in range(N)]\
+[x2[k]>=-0.1 for k in range(N)]


prob = cp.Problem(cp.Minimize(obje),cons)
x1_0.value=0
x2_0.value=0
x3_0.value=0
x4_0.value=0

prob.solve()




####################################################################################
#                                                                                  #
#                        Construction of a delta-net                               #
#                                                                                  #
####################################################################################


#deta net
n_of_sample=100
q1=int(n_of_sample/2)*[1]
q2=int(n_of_sample/2)*[2]

alpha_1=np.array(q1+q2)
delta=1/n_of_sample*2
q=list(np.arange(delta,1+delta,delta))

alpha_2=np.array(2*q)

    
####################################################################################
#                                                                                  #
#           Reconstruction of a feasible control and trajectory                    #
#                                                                                  #
####################################################################################    
    

def generate_basis(x):
    basis=[]
    for i in range(n_of_sample):
        a=[alpha_1[i],alpha_2[i]]
        basis.append(f(x,a))
    return basis


# aux function
def aux(x,n):
    z=[]
    for i in range(n):
      z.append(x)
    z=np.array(z)
    return(z)
    

z=[]
for i in range(N):
    x=[x1.value[i],x2.value[i],x3.value[i],x4.value[i]]
    basis=generate_basis(x)
    basis=np.array(basis)
    approx=np.array([b1.value[i],b2.value[i],b3.value[i],b4.value[i]])
    
    nearest=(aux(approx,n_of_sample)-basis)**2
    nearest=np.sum(nearest,axis=1)
    index=np.argmin(nearest)
    
    v=basis[index]
    z.append(v)
    max_basis=z

a1=[]
a2=[]
for i in range(N):
    a1.append(-max_basis[i][3]/max_basis[i][1])

for i in range(N):
    a2.append(-max_basis[i][1]*(c1+c2*a1[i]**2))

tt=np.arange(0,1,1/N)
plt.plot(tt,a1,label=r'$u_1$',color='r')
plt.plot(tt,a2,label=r'$u_2$',color='b')
plt.xlabel('t',fontsize=15)
plt.legend(fontsize=15,loc='lower left')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.yticks(np.arange(0,max(a1)+0.5,0.5))
plt.savefig("control.png",format='png',dpi=1200)

    
    
  
####################################################################################
#                                                                                  #
#           Plot state trajectory with new control.                                #
#                                                                                  #
####################################################################################   
 
def f(x,a1,a2):
return [x[1],1/(1+3*a1**2)*a2,x[3],-a1/(1+3*a1**2)*a2]


x_initial=[0,0,0,0]

traj=[]

traj.append(x_initial)

traj_aux=x_initial

for i in range(N):
    
    traj_aux=traj_aux+dt*np.array(f(traj[i],a1[i],a2[i]))
    
    traj.append(traj_aux)
    
    
x1_=[]
for i in range(len(traj)):
    x1_.append(traj[i][0])

tt=np.arange(0,1+1/N,1/N)
plt.plot(tt,x1_,label='$x_1$ (Lagrangian)',color='r')
plt.xlabel('t',fontsize=15)
plt.legend(fontsize=15,loc='lower right')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.yticks(np.arange(0,max(x1_)+0.02,0.02))
plt.savefig("x1.png",format='png',dpi=1200)
    
      
x2_=[]
for i in range(len(traj)):
    x2_.append(traj[i][1])

plt.plot(tt,x2_,label='$x_2$ (Lagrangian)',color='r')
plt.xlabel('t',fontsize=15)
plt.legend(fontsize=15,loc='lower right')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.yticks(np.arange(0,0.12,0.02))
plt.savefig("x2.png",format='png',dpi=1200)


x3_=[]
for i in range(len(traj)):
    x3_.append(traj[i][2])

plt.plot(tt,x3_,label='$x_3$ (Lagrangian)',color='r')
plt.xlabel('t',fontsize=15)
plt.legend(fontsize=15,loc='lower left')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.yticks(np.arange(-0.1,max(x3_)+0.02,0.02))
plt.savefig("x3.png",format='png',dpi=1200)


x4_=[]
for i in range(len(traj)):
    x4_.append(traj[i][3])

plt.plot(tt,x4_,label='$x_4$ (Lagrangian)',color='r')
plt.xlabel('t',fontsize=15)
plt.legend(fontsize=15,loc='lower left')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.yticks(np.arange(-0.15,max(x4_)+0.05,0.05))
plt.savefig("x4.png",format='png',dpi=1200)
