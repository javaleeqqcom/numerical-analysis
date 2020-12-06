#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import math as M

#返回x的n位十进制有效数字近似值
def rn(x:float,n:int):
    if x==0.0: return 0.0
    p= M.log(M.fabs(x),10)
    lg=int(M.ceil(p))
    if lg>=n:
        base =int(pow(10,int(lg)-n))
        return float(round(x/base)*base)
    else:
        return round(x, n-lg)

#对ndarray多维数组对象的每个元素a，代入到fun(a,*para)函数，并将结果值按原数组排列返回一个新数组。
def for_exert(arr: np.ndarray,fun,*para):
    L=arr.flatten() #不会改变原数组arr！
    for i,a in enumerate(L):
        L[i]=fun(a,*para)
    return L.reshape(arr.shape)

A=[0.729,0.81,0.9,
   1,1,1,
   1.331,1.21,1.1]
A=np.array(A).reshape((3,3))
b=[0.8338,0.8338,1.0]
b= np.array(b).reshape((3,1)) #结果列向量b

# Doolittle分解 A=L·U
def doolittle(A:np.ndarray):
    L=np.eye(A.shape[0])   # L初始为单位矩阵
    U=A.copy()
    for j in range(A.shape[1]-1):
        for i in range(j+1,A.shape[0]):
            k=U[i,j]/U[j,j]
            Ai_=U[i,j:] - U[j,j:]*k
            U[i,j:] = for_exert( Ai_, rn,4)
            L[i,j]=  rn( k ,4) #保留4位有效数字
        # U=for_exert(U,rn,4) #保留4位有效数字
        print('j=',j,'U=\n', U)
    print('L=',L)
    _A=np.dot(L,U)  #比较 L·U 的近视计算结果与A的差别
    print("L·U=\n",_A)
    print("A=\n",A)

def gaussian_elimination_method(A:np.ndarray,b:np.ndarray,col_min:bool):
    Ab=np.concatenate((A,b),axis=1)
    L=np.eye(A.shape[0])   # L初始为单位矩阵
    print('Ab=\n',Ab)
    for j in range(A.shape[1]-1):
        if col_min:
            col_j_arr_i=[(abs(a),i) for i,a in enumerate(A[:,j].flatten())]
            max_i=max(col_j_arr_i[j:])[1]
            if max_i!=j:
                Ab[[j,max_i], :] = Ab[[max_i,j], :] # j交换矩阵Ab的行max_i和行j
                print("swap_row(%d,%d),then Ab=\n"%(j,max_i),Ab)
        for i in range(j+1,A.shape[0]):
            k=Ab[i,j]/Ab[j,j]
            k=rn( k ,4) #保留4位有效数字
            Aj_mul_k=Ab[j,j:]*k
            Aj_mul_k=for_exert(Aj_mul_k,rn,4) #保留4位有效数字
            Ai_=Ab[i,j:] - Aj_mul_k
            Ai_=for_exert( Ai_, rn,4) #保留4位有效数字
            Ab[i,j:] = Ai_
            L[i,j]= k
        print('j=%d'%(j),'\t,Ab=\n', Ab)
    print('L=\n',L)
    x=np.zeros(b.shape[0]).reshape(b.shape)
    for i in range(b.shape[0]-1,-1,-1):
        if Ab[i,i]!=0.0:
            xi=(Ab[i,-1]-np.dot(Ab[i,:-1],x))/Ab[i,i]
            xi=for_exert(xi,rn,4)   #保留4位有效数字
            x[i,0]= xi[0]
        else:
            x[i,0]=np.NAN
    print('x=\n',x)
    return x

print("高斯消元法")
x_1=gaussian_elimination_method(A,b,False)

print("\n高斯列主元消元法")
x_2=gaussian_elimination_method(A,b,True)

x_ac= np.linalg.solve(A, b)
print('x_ac=',x_ac.flatten() )
print('A·x_ac-b=\n',np.dot(A,x_ac)-b)
print('A·x_1-b=\n',np.dot(A,x_1)-b)
print('A·x_2-b=\n',np.dot(A,x_2)-b)
print((x_ac-x_1))
print((x_ac-x_2))

print("1范数")
print('d1=%f'%(sum(abs(x_ac-x_1))))
print('d2=%f'%(sum(abs(x_ac-x_2))))

print("2范数")
print('d1=%f'%(M.sqrt(sum((x_ac-x_1)**2))))
print('d2=%f'%(M.sqrt(sum((x_ac-x_2)**2))))

print("无穷范数")
print('d1=%f'%(max(x_ac-x_1)))
print('d2=%f'%(max(x_ac-x_2)))