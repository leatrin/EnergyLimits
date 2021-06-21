import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def sum(table) : 
    s = 0
    for i in table : 
        s+=i 
    return s

def growth_rate(x, y,s = 1, a = 1  ) : 
    # vérifier que le pas de x est bien de 1, sinon interpoler linéairement
    x_min, x_max =x.min(),x.max()
    X = np.arange(x_min, x_max,1 )
    Y=  []
    i= 0
    s1, s2  = s// 2, s- s//2
    a1, a2 = a//2, a- a//2
    r, m, x_r, x_m = [], [], [], []

    for val in X :
        if int(val) in list(x) : 
            Y.append(y[i])
            i+=1
        else : 
            slope =( y[i]-y[i-1])/(x[i]-x[i-1])
            origin = y[i]-slope*x[i]
            ynew= slope*val+origin
            Y.append(ynew)

    # compute r_n(s)

    for n in range(0,s1) : 
        x_r.append(x_min+n)
        rn= (1/s)*((Y[n+s]-Y[n])/Y[n+s])*100
        r.append(rn)
    for n in range(s1, len(X)-s2) : 
        x_r.append(x_min+n)
        rn= (1/s)*((Y[n+s2]-Y[n-s1])/Y[n-s1])*100
        r.append(rn)
    for n in range (len(X)-s2, len(X)) :
        x_r.append(x_min+n)
        rn= (1/s)*((Y[n]-Y[n-s])/Y[n-s])*100
        r.append(rn)
    # compute avg 
    for n in range(a1, len(x_r)-a2) : 
        x_m.append(np.array(x_r).min()+n)
        mn = sum(r[n-a1:n+a2])*(1/a)
        m.append(mn)

    
    return x_m, m



    
