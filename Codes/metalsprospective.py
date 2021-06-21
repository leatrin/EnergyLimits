import numpy as np
import matplotlib.pyplot as plt
def ind(x,a,b) : 
    if a<x<b :
        return 1
    else : 
        return 0


def fADC(x, a, d, x1, x2) :
    c=a*x1
    e = c-d*x2
    return a*x*ind(x,0,x1)+c*ind(x, x1, x2)+ (d*x+e)*ind(x, x2, 10**10)

fADC = np.vectorize(fADC)

def ADC_GDP(metal = 'Alu') : 
    GDPc = np.linspace(0,45, 500)
    low = fADC(GDPc, 15, -15/26, 23,25 )
    med = fADC(GDPc, 17, -15/26, 17,30 )
    high = fADC(GDPc, 22, -15/26, 15,35 )

    plt.plot(GDPc, low, label='low')
    plt.plot(GDPc, med, label='med')
    plt.plot(GDPc, high, label='high')
    plt.xlabel("GDP per capita (k$)")
    plt.ylabel("ADC per capita (t)")
    plt.xlim(0,45)
    plt.ylim(0,25)
    
    plt.legend()
    return None



