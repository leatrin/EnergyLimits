import numpy as np 
from scipy.stats import linregress
from math import exp, log, pi, sqrt, cosh
from scipy.optimize import curve_fit
import math as math
import sympy as sy
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math as math

def l2_loss(y_real, y_predicted):
    l2_err = np.sqrt((y_real-y_predicted)**2)
    return np.sum(l2_err)


######### LINEAR MODEL #######################
def lin_fun_scalar(x, y0 = 0, r = 1): 
    return y0 + r*x

lin_fun = np.vectorize(lin_fun_scalar)

def linReg(x,y) :
    """
    Returns r, y0
    """
    a, b, cor, p_value, std_err = linregress(x, y)
    r, y0 = a, b
    return r, y0

def linLoss(x,y,r,y0) : 
    return l2_loss(y, lin_fun(x,y0,r))

######### EXPONENTIAL MODEL #######################

def exp_fun_scalar(x, y0 = 0, r = 1): 
    return y0*exp(r*x)

exp_fun = np.vectorize(exp_fun_scalar)

def expReg(x,y) :

    a, b, cor, p_value, std_err = linregress(x, np.log(y))
    y0 = exp(b)
    r = a
    return r, y0

def expLoss(x,y,r,y0) : 
    return l2_loss(y, exp_fun(x,y0,r))


######### LOGISTIC MODEL #######################

def log_fun_scalar(x, K = 10, dt = 50, tm=25): 
    return K/(1+exp((-log(81)/dt)*(x-tm)))

log_fun = np.vectorize(log_fun_scalar)

def initialLogParams(x, y) :
    """
    Returns K, dt, tm
    """
    K = np.max(y)
    m= np.array([abs(0.5*K-y[i]) for i in range(len(y))]).argmin()
    m1 = np.array([abs(0.1*K-y[i]) for i in range(len(y))]).argmin()
    m9 = np.array([abs(0.9*K-y[i]) for i in range(len(y))]).argmin()
    tm = x[m]
    dt = x[m9]-x[m1]

    return K, dt, tm

def logReg (x, y, Dy = None, bounds = [0, 90000]) :
    """
    Returns K, dt, tm
    """
    K0, dt0, tm0 = initialLogParams(x,y)
    
    popt, pcov = curve_fit(
        f = log_fun,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=(K0, dt0, tm0),      # initial value of the parameters
        bounds = bounds,
        sigma=Dy
    )
    K, dt, tm = popt
    return K, dt, tm

######### S-CURVE MODEL #######################


def s_model_scal(x, G_i=1, E_i=1, A=1, a1=1, a2=1, a3=1):
    return (E_i+ A*(math.exp(a1*(x-G_i))- math.exp(-a3*(x-G_i)))/(2*math.cosh(a2*(x-G_i))))

s_model=np.vectorize(s_model_scal)


def f(A, eth_1, eth_2, eth_3) :
    return math.tanh(eth_1/A)*math.tanh(eth_2/A)-eth_3

def find_A(A_min, A_max, eth_1, eth_2, eth_3, epsilon=10**-3) :
    """
    This function based on dichotomy allows me to estimate the value of A variable (cf article)
    """
    while A_max-A_min > epsilon :
        A = (A_min+A_max)/2
        if f(A, eth_1, eth_2, eth_3) ==0 :
            return A  
        elif f(A_min, eth_1, eth_2, eth_3)*f(A_max, eth_1, eth_2, eth_3)< 0:
            A_max =A
        else :
            A_min = A

    return (A_min+A_max)/2

def initialSParams(G, E, deg = 6, step = 0.1) : 
    """
    Returns G_i, E_i, A, a1, a2, a3
    """
    id_i = E.argmax()//2
    E_i = E[id_i]
    G_i = G[id_i]

    # zero-growth point 
    G_z = G[E.argmax()]
    E_z = E.max()

    # Take-off point 
    G_t = G[id_i//3]
    E_t = E[id_i//3]

    # slope before the take-off point kl
    kl = (E_t-E[0])/(G_t-G[0])
    # slope between take-off point and the zero-growth point ki
    ki = (E_z-E_t)/(G_z-G_t)
    # slope after the zero-growth point kv
    n = len(G)-1
    kv = (E[n-1]-E_z)/(G[n-1]-G_z)

    # parameters
    eth_1 = 0.5*(ki+kl+kv)*(G_z-G_i)
    eth_2 = 0.5*(kl+2*ki-kv)*(G_z-G_i)
    eth_3 = (kl+ki+kv)/(kl+2*ki-kv)
    
    A = find_A(1,10, eth_1, eth_2, eth_3)

    alpha_1 = (kl+ki+kv)/(2*A)
    alpha_2 = (kl+2*ki-kv)/(2*A)
    alpha_3 = (-kl+2*ki-kv)/(2*A)


    return G_i, E_i, A, alpha_1, alpha_2, alpha_3

def sReg(xdata, ydata, Dy=None,  bounds = [-350000, 350000]):
    G_i, E_i, A, a1, a2, a3= initialSParams(xdata, ydata, deg=6, step = 0.1)
    p0 = ( G_i, E_i, A, a1, a2, a3)
    print(p0)
    popt, pcov = curve_fit(
        f = s_model,       # model function
        xdata=xdata,   # x data
        ydata=ydata,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
        sigma=Dy
    )
    G,E, A, a1, a2, a3  = popt

    return G,E, A, a1, a2, a3

############# lin then exp ################
def ind (x,a,b) : 
    if x<=b and x>=a : 
        return 1
    else : 
        return 0


def lin_exp_scalar (x, a, xlim, r, y0): 
    f= (a*(x-xlim)+ y0* math.exp(xlim*r))*ind(x, -10**10, xlim) + y0*math.exp(r*x)*ind(x, xlim, 10**10)
    return f

lin_exp = np.vectorize(lin_exp_scalar)

def linExpReg (x, y, p0, bounds = [0, 90000]) :
    """
    Returns a, xlim, r, y0
    """
    popt, pcov = curve_fit(
        f = lin_exp,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    a, xlim, r, y0 = popt
    return a, xlim, r, y0


####### lin then pow #########
def lin_pow_scalar(x, a, xlim, r, y0) :
    b= y0*(xlim**r) - a *xlim 
    return (a*x+b)*ind(x, -10**10, xlim) + y0*(x**r)*ind(x, xlim, 10**10)

lin_pow =np.vectorize(lin_pow_scalar)

def linPowReg (x, y, p0, bounds = [0, 90000]) :
    """
    Returns a, xlim, r, y0
    """
    popt, pcov = curve_fit(
        f = lin_pow,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    a, xlim, r, y0 = popt
    return a, xlim, r, y0

####### exp lin lin #########
def exp_lin_lin_scalar(x, y0, r, a, m, t1, t2) :
    b= y0*math.exp(r*t1)-a*t1
    p = (a-m)*t2+b
    return y0*math.exp(r*x)*ind(x, -10**10, t1) + (a*x+b)*ind(x, t1, t2) + (m*x+p )*ind(x, t2, 10**10)

exp_lin_lin = np.vectorize(exp_lin_lin_scalar)

def expLinLin (x, y, p0, bounds = [0, 90000]) :
    """
    Returns y0, r, a, m, t1, t2
    """
    popt, pcov = curve_fit(
        f = exp_lin_lin,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    y0, r, a, m, t1, t2 = popt
    return y0, r, a, m, t1, t2

######## lin pow lin ############

def lin_pow_lin_scalar(x, a,y0,r,m, x1, x2) :
    b= y0*(x1**r)-a*x1
    p = y0*(x2**r)-m*x2
    return (a*x+b)*ind(x, -10**10, x1) + y0*(x**r)*ind(x, x1, x2) + (m*x+p)*ind(x, x2, 10**10)

lin_pow_lin = np.vectorize(lin_pow_lin_scalar)

def linPowLin (x, y, p0, bounds = [0, 90000]) :
    """
    Returns a,y0,r,m, x1, x2
    """
    popt, pcov = curve_fit(
        f = lin_pow_lin,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    a,y0,r,m, x1, x2 = popt
    return a,y0,r,m, x1, x2

############ hyp lin lin ########### 

def hyp_lin_lin_scalar(x, b,c,d,f,x1,x2) :
    e = (b/(-x1+c)) - d*x1 
    g = (d-f)*x2 + e
    if x <= x1 : 
        return (b/(c-x))*ind(x, -10**10, x1) 
    elif x1 < x <= x2 :
        return (d*x+e)
    else : 
        return (f*x+g)

hyp_lin_lin = np.vectorize(hyp_lin_lin_scalar)

def hypLinLin (x, y, p0, bounds = [0, 90000]) :
    """
    Returns b,c,d,f,x1,x2
    """
    popt, pcov = curve_fit(
        f = hyp_lin_lin,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    b,c,d,f,x1,x2= popt
    return b,c,d,f,x1,x2

####### exp lin lin lin lin ######
def expllll(x, y0, r, a,c,e,g,x1,x2,x3,x4) :
    b = y0*math.exp(r*x1)-a*x1
    d = (a-c)*x2+b
    f = (c-e)*x3+d
    h = (e-g)*x4+f 
    return y0*exp(r*x)*ind(x, -10**10, x1) + (a*x+b)*ind(x, x1, x2)+(c*x+d)*ind(x, x2, x3)+ (e*x+f)*ind(x, x3, x4)+(g*x+h) * ind(x, x4, 10**10)

expllll = np.vectorize(expllll)

def expLLLL (x, y, p0, bounds = [-90000, 90000]) :
    """
    Returns y0, r, a,c,e,g,x1,x2,x3,x4
    """
    popt, pcov = curve_fit(
        f = expllll,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    y0, r, a,c,e,g,x1,x2,x3,x4= popt
    return y0, r, a,c,e,g,x1,x2,x3,x4


####### hyp lin lin lin lin ########

def hypllllscalar(x, b,c,d,f,h,k,x1,x2, x3, x4) :
    e = (b/(-x1+c)) - d*x1 
    g = (d-f)*x2 + e
    j= (f-h)*x3 + g
    l = (h-k)*x4 +j
    if x < x1 : 
        return (b/(c-x))*ind(x, -10**10, x1) 
    else :
        return (d*x+e)*ind(x, x1,x2) +(f*x+g)*ind(x, x2, x3)+ (h*x+j)*ind(x, x3, x4) + (k*x+l)*ind(x, x4, 10**10)
    
hypllll = np.vectorize(hypllllscalar)

def hypLLLL (x, y, p0, bounds = [-90000, 90000]) :
    """
    Returns b,c,d,f,h,k,x1,x2, x3, x4
    """
    popt, pcov = curve_fit(
        f = hypllll,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    b,c,d,f,h,k,x1,x2, x3, x4 = popt
    return b,c,d,f,h,k,x1,x2, x3, x4

###### Lin lin lin lin  #####

def llll(x, a,b, c, e, g, x1, x2, x3) :
    d = (a-c)*x1 + b
    f = (c-e)*x2 +d
    h = (e-g)*x3 + f
    return (a*x+b)*ind(x, -10**10, x1)+ (c*x+d)*ind(x, x1, x2)+ (e*x+f) *ind(x, x2, x3)+ (g*x+h)*ind(x, x3, 10**10)

llll = np.vectorize(llll)
def LLLL (x, y, p0, bounds = [-90000, 90000]) :
    """
    Returns a,b, c, e, g, x1, x2, x3
    """
    popt, pcov = curve_fit(
        f = llll,       # model function
        xdata=x,   # x data
        ydata=y,   # y data
        p0=p0,      # initial value of the parameters
        bounds = bounds,
    )
    a,b, c, e, g, x1, x2, x3 = popt
    return a,b, c, e, g, x1, x2, x3