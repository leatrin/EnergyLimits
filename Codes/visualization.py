from regressions import linReg, lin_fun, exp_fun, expReg, log_fun, logReg, s_model, sReg
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np 


def l2_loss(y_real, y_predicted):
    l2_err = np.sqrt((y_real-y_predicted)**2)
    return np.sum(l2_err)

def test_models(x, y, x_title = None, y_title = None, xmin = 0, xmax = 3000, logbounds = [0, 90000], sbounds = [-350000, 350000]) :

    # Linear Model 
    rl, y0l = linReg(x, y)
    yLin = lin_fun(x, r=rl, y0=y0l)
    lin_loss = l2_loss(y, yLin)
    # Exponential model
    re, y0e = expReg(x,y)
    yExp = exp_fun(x,y0e, re)
    exp_loss = l2_loss(y, yExp)
    # Logistic Model
    K, dt, tm = logReg(x,y, bounds = logbounds)
    yLog = log_fun(x, K, dt, tm)
    log_loss =  l2_loss(y, yLog)
    # S-Model 
    G_i, E_i, A, alpha_1, alpha_2, alpha_3 = sReg(x,y, bounds = sbounds)
    yS = s_model(x, G_i, E_i, A, alpha_1, alpha_2, alpha_3)
    s_loss = l2_loss(y, yS)

    sns.set_theme()
    x_plot= np.linspace(xmin  , xmax, 200)
    plt.figure(figsize=(16,10))
    plt.subplot(221) 
    plt.scatter(x,y, s=3)
    plt.plot(x_plot,lin_fun(x_plot, r=rl, y0=y0l) ) 
    plt.title('Linear model:'+str(lin_loss))
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.subplot(222)
    plt.scatter(x,y, s=3)
    plt.plot(x_plot,exp_fun(x_plot, r=re, y0=y0e) ) 
    plt.title('Exponential model:'+str(exp_loss))
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.subplot(223)
    plt.scatter(x,y, s=3)
    plt.plot(x_plot,log_fun(x_plot, K, dt, tm)) 
    plt.title('Logistic model:'+str(log_loss))
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    plt.subplot(224)
    plt.scatter(x,y, s=3)
    plt.plot(x_plot,s_model(x_plot, G_i, E_i, A, alpha_1, alpha_2, alpha_3))
    plt.title('S model:'+str(s_loss))
    plt.xlabel(x_title)
    plt.ylabel(y_title)

    return None 




