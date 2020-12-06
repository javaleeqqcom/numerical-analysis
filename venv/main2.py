#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import scipy as sp
import fractions as fc
import matplotlib.pyplot as plt

alg=np.linalg
x=np.arange(0,4.1,0.5)
y=np.array([4  ,2.927, 2.470 ,2.393 ,2.54 ,2.829 ,3.198 ,3.621 ,4.072])
# x=np.array([fc.Fraction(int(10*a),10) for a in x])
# y=np.array([fc.Fraction(int(1000*a),1000) for a in y])
print( x,'\n',y)
engx=np.exp(-x)
l=np.array([x,engx])
G=l.dot(l.transpose())
print(G)
d=l.dot(y)
γ=alg.solve(G,d)

plt.plot(x,y, '*',label='xi-yi')  # 画线并添加图例legend
X=np.arange(-0.1,4.1,0.01)
lX=np.array([X,np.exp(-X)])
φ=γ.dot(lX)
plt.plot(X, φ, '-',label='φ(x)')  # 画线并添加图例legend
plt.xlabel('x')  # 给 x 轴添加坐标轴信息
plt.ylabel('y')  # 给 y 轴添加坐标轴信息
plt.legend()
plt.title('fitting for problem 4.17')  # 添加图片标题
plt.show()
