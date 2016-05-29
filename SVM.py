# -*- coding: utf-8 -*-
"""
Created on Sun May 08 14:37:32 2016

@author: jurcol
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lp
import sys
from mpl_toolkits.mplot3d import Axes3D

# input the 2D data in 2 arrays for 2 classes
n = 100
x = np.zeros((n,2))
y = np.zeros(n)
x[0:n/2,:] = np.random.multivariate_normal([1,1],[[1,0],[0,1]],n/2)
y[0:n/2] = 1
x[n/2:n,:] = np.random.multivariate_normal([10,10],[[1,0],[0,1]],n/2)
y[n/2:n] = -1
plt.plot(x[0:n/2,0],x[0:n/2,1],'.',color = 'green')
plt.plot(x[n/2:n,0],x[n/2:n,1],'*',color = 'red')

def compute_kernel1(m,n):
    sum_p = 0
    for i in range(m.size):
        sum_p = sum_p + m[i] * n[i]
    return sum_p

def lagrange(H, c, A, b):
    c = c.reshape((H.shape[0], 1))
    b = b.reshape((A.shape[0], 1))
    inv_H =  np.linalg.pinv(H)
    AHA = np.dot(np.dot(A, inv_H), A.T)
    inv_AHA = np.linalg.pinv(AHA)
    A_inv_H = np.dot(A, inv_H)
    G = inv_H - np.dot(np.dot(A_inv_H.T, inv_AHA), A_inv_H)
    B = np.dot(inv_AHA, A_inv_H)
    x = np.dot(B.T, b) - np.dot(G, c)
    return x

def compute_kernel(m,n):
    return math.exp(-0.5*np.linalg.norm(m-n))
    
def compute_value(x,b,vari,data,y):
    sum_d = 0
    for i in range(y.size):
        sum_d = sum_d + math.exp(-0.5 * np.linalg.norm(x[i,:]-data)) * vari[i] * y[i]
    return sum_d-b
    
c = -1 * np.ones(n).reshape((n,1))
H = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        H[i,j] = compute_kernel(x[i,:],x[j,:])*y[i]*y[j]
Ae = y[:]
be = 0.
Ai = np.eye(n)
bi = np.zeros(n)
err = 1e-12

def find_minimum(q):
    temp1 = sys.float_info.max
    index4 = -1
    for i in range(1,q.size):
        if q[i] < temp1:
            temp1 = q[i]
            index4 = i
    return temp1,index4

def compute_sum(x,y,vari):
    sumw = np.zeros(2).reshape((1,2))
    for i in range(n):
        sumw += x[i,:] * y[i] * vari[i]
    return sumw

def compute_alpha(A,b,vari,dk):
    min_alpha = 1
    index3 = -1
    for i in range(b.size):
        if np.dot(A[i,:],dk) < err :
            temp1 = (b[i]-np.dot(A[i,:],vari))/np.dot(A[i,:],dk)
            if temp1 < min_alpha:
                min_alpha = temp1
                index3 = i
    return min_alpha, index3

vari = np.ones(n).reshape((n,1))

Ak = []
bk = []
Akn = Ai.tolist()
bkn = bi.tolist()
Ak.append(Ae)
bk.append(be)

dele = 0
for i in range(n):
    if np.dot(Ai[i,:],vari)-bi[i] <= err:
        Ak.append(Ai[i,:])
        bk.append(bi[i])
        del Akn[i-dele]
        del bkn[i-dele]
        dele = dele + 1

while(True):
    gk = np.dot(H, vari) + c
    dk = lagrange(H, gk, np.array(Ak),np.array(bk))
    temp = np.dot(np.dot(np.array(Ak),lp.pinv(H)),np.array(Ak).T)
    Bk = np.dot(np.dot(lp.pinv(temp),np.array(Ak)),lp.pinv(H))
    if  np.linalg.norm(dk) >= err :
        if np.all(np.dot(np.array(Akn).reshape((len(Akn),n)),dk) > -err):
            alpha = 1
            vari = vari + dk
        else:
             min_alpha = compute_alpha(np.array(Akn),np.array(bkn),vari,dk)
             alpha = min_alpha[0]
             vari = vari + alpha * dk
             if min_alpha[1] >= 0:
                 Ak.append(Akn[min_alpha[1]])
                 bk.append(bkn[min_alpha[1]])
                 del Akn[min_alpha[1]]
                 del bkn[min_alpha[1]]
    else:
        lamb = np.dot(Bk,gk)
        lambd = find_minimum(lamb)
        if lambd[0] >= -err:
            break
        else:
            Akn.append(Ak[lambd[1]])
            bkn.append(bk[lambd[1]])
            del Ak[lambd[1]]
            del bk[lambd[1]]
          
index_b = 0
for i in range(vari.size):
    if vari[i] >  err:
        break
    index_b = i
sum_b = 0
for i in range(n):
    sum_b = sum_b + vari[i] * y[i] * math.exp(-0.5*np.linalg.norm(x[i,:]-x[index_b,:]))
b = sum_b - y[index_b]
#print vari
#print b
#print compute_value(x,b,vari,x[index_b,:],y)

x1= np.linspace(-2,12,100)
x2 = np.linspace(-2,12,100)
[x1,x2] = np.meshgrid(x1,x2)
#print x1,x2
Y = np.empty((100,100))
for i in range(100):
    for j in range(100):
        data = np.empty(2)
        data[0] = x1[i,j]
        data[1] = x2[i,j]
        Y[i,j] = compute_value(x,b,vari,data,y)
plt.contour(x1,x2,Y,levels = np.linspace(-1,1,7))
plt.show()


