# vaccinazioni sono partite il 27 Dicembre 2020

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import math
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.statespace.sarimax as sarima
from sklearn.metrics import mean_squared_error
import itertools

#importazione dati
data = pd.read_csv(r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\somministrazioni-vaccini.csv')
date_ordinate = []
for i in pd.date_range(start="2020-12-27",end="2021-03-18",freq='D'):
    date_ordinate.append(i)
#print(date_ordinate[0])

#calcolo vaccinazioni totali giornaliere dal dataset
vaccini_ordinate = []

for i in pd.date_range(start="2020-12-27",end="2021-03-18",freq='D'):
    vaccini_ordinate.append(0)
#print(len(vaccini_ordinate))
#print(data['data_somministrazione'][125])
#print(data['data_somministrazione'][125]==str(date_ordinate[0]))

pippo=0
for i in pd.date_range(start="2020-12-27",end="2021-03-18",freq='D'):
    for j in range(len(data)):
        
        if(data['data_somministrazione'][j]==str(i)):
            vaccini_ordinate[pippo]+=data['totale'][j]
    pippo+=1
print(vaccini_ordinate)

#ora triple exp smoothing

size = int(len(vaccini_ordinate)*0.8)
vaccini_ordinate_train=vaccini_ordinate[0:size]
vaccini_ordinate_test=vaccini_ordinate[size:len(vaccini_ordinate)]
#print(vaccini_ordinate_train)
#print(vaccini_ordinate_test)

fit_triple1= ExponentialSmoothing(vaccini_ordinate_train, trend= "add", seasonal= "add", seasonal_periods=7).fit()
fcast_triple1 = fit_triple1.forecast(len(vaccini_ordinate)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend'])
fit_triple2 = ExponentialSmoothing(vaccini_ordinate_train, trend= "add", seasonal= "mul", seasonal_periods=7).fit()
fcast_triple2 = fit_triple2.forecast(len(vaccini_ordinate)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend'])
fit_triple3 = ExponentialSmoothing(vaccini_ordinate_train, trend= "mul", seasonal= "add", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple3 = fit_triple3.forecast(len(vaccini_ordinate)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend'])
fit_triple4 = ExponentialSmoothing(vaccini_ordinate_train, trend= "mul", seasonal= "mul", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple4 = fit_triple4.forecast(len(vaccini_ordinate)-size)    # PROBABILMENTE LA MIGLIORE
#print(fcast_triple1)
mse1 = ((fcast_triple1 - vaccini_ordinate_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = add and seasonal = add is {}'.format(round(np.sqrt(mse1), 2)))
mse2 = ((fcast_triple2 - vaccini_ordinate_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = add and seasonal = mul is {}'.format(round(np.sqrt(mse2), 2)))
mse3 = ((fcast_triple3 - vaccini_ordinate_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = mul and seasonal = add is {}'.format(round(np.sqrt(mse3), 2)))
mse4 = ((fcast_triple4 - vaccini_ordinate_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = mul and seasonal = mul is {}'.format(round(np.sqrt(mse4), 2)))

xValues=[]
for i in range(size,len(vaccini_ordinate)):
    xValues.append(i)


fig,axs=plt.subplots(2,2)
axs[0,0].plot(vaccini_ordinate, marker='o', color='black')
axs[0,0].plot(fit_triple1.fittedvalues, marker='o', color='blue')
axs[0,0].plot(xValues,fcast_triple1, marker='o', color='cyan')
plt.sca(axs[0, 0])
plt.xticks([0,15,30,45,60,75],["27 Dic", "11 Gen", "26 Gen", "10 Feb", "25 feb", "12 Mar"])
axs[0,0].set_title("add add")

axs[0,1].plot(vaccini_ordinate, marker='o', color='black')
axs[0,1].plot(fit_triple2.fittedvalues, marker='o', color='red')
axs[0,1].plot(xValues,fcast_triple2, marker='o', color='cyan')
plt.sca(axs[0, 1])
plt.xticks([0,15,30,45,60,75],["27 Dic", "11 Gen", "26 Gen", "10 Feb", "25 feb", "12 Mar"])
axs[0,1].set_title("add mul")

axs[1,0].plot(vaccini_ordinate, marker='o', color='black')
axs[1,0].plot(fit_triple3.fittedvalues, marker='o', color='green')
axs[1,0].plot(xValues,fcast_triple3, marker='o', color='cyan')
plt.sca(axs[1, 0])
plt.xticks([0,15,30,45,60,75],["27 Dic", "11 Gen", "26 Gen", "10 Feb", "25 feb", "12 Mar"])
axs[1,0].set_title("mul add")

axs[1,1].plot(vaccini_ordinate, marker='o', color='black')
axs[1,1].plot(fit_triple4.fittedvalues, marker='o', color='brown')
axs[1,1].plot(xValues,fcast_triple4, marker='o', color='cyan')
plt.sca(axs[1, 1])
plt.xticks([0,15,30,45,60,75],["27 Dic", "11 Gen", "26 Gen", "10 Feb", "25 feb", "12 Mar"])
axs[1,1].set_title("mul mul")
fig.suptitle('TripleExpSmoothing')
plt.show()

# ora ARIMA

def sarima_grid_search(y,seasonal_period):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2],seasonal_period) for x in list(itertools.product(p, d, q))]
    
    mini = float('+inf')
	
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sarima.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
                results = mod.fit()  
                if results.aic < mini:
                    mini = results.aic
                    param_mini = param
                    param_seasonal_mini = param_seasonal


            except:
                continue
    print('The set of parameters with the minimum AIC is: SARIMA{}x{} - AIC:{}'.format(param_mini, param_seasonal_mini, mini))


# Call this function after pick the right(p,d,q) for SARIMA based on AIC
#         
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test,y_train):
    # fit the model 
    mod = sarima.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    
    # results.plot_diagnostics(figsize=(16, 8))     # plot di ulteriori dati
    # plt.show()  
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    pred = results.get_prediction(start=pred_date, dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))
    '''
    ax = plt.plot(y, label='observed')  # modificato da ax = y.plot(label='observed')
    plt.plot(y_forecasted, ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))   # QUI ERRORE
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Data')
    ax.set_ylabel('nuovi_positivi')
    plt.legend()
    plt.show()
    '''
    # (1) If we are forecasting x(T+1), x(T+2)... x(T+h), using information up to period T, 
    # we may have to do it recursively by using forecast of x(T+1) in order to forecast x(T+2) and so on.
    # (2) If we are forecasting x(T+1) using information up to time T, then forecasting x(T+2) using information up to period T+1 and so on, 
    # we may have to wait each step until we have updated information (or pretend we are waiting).
    # Call (1) "Forecast with a FIXED INFORMATION SET" by recursively using predicted values if needed. 
    # This is what EViews calls "dynamic forecasting".
    # Call (2) "Forecast with a MOVING ESTIMATION SAMPLE" one-step-ahead only (or, to be more general, with a fixed h-step-ahead). 
    # This is what EViews calls "static forecasting"

    pred_dynamic = results.get_prediction(start=pred_date, dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))
    '''
    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Data')
    ax.set_ylabel('nuovi_positivi')

    plt.legend()
    plt.show()
    '''
    # nostro plot
    fig, ax = plt.subplots()
    asse_x=np.arange(0,len(y_train),1)
    asse_x2=np.arange(len(y_train)+1,len(y),1)
    ax.plot(asse_x,y_train, label='X_train')
    ax.plot(asse_x2,y_to_test, label='X_test')
    ax.plot(asse_x2,y_forecasted, label='y_forecasted')
    ax.plot(asse_x2,y_forecasted_dynamic, label='y_forecasted_dynamic')
    plt.legend(loc='upper left')
    plt.title('SARIMAX')
    plt.ylabel('Vaccini somministrati')
    plt.xlabel('Data')
    plt.xticks([0,15,30,45,60,75],["27 Dic", "11 Gen", "26 Gen", "10 Feb", "25 feb", "12 Mar"])
    plt.show()
    return (results)
    

X = vaccini_ordinate
# X = X.asfreq('W')

size = int(len(X)*0.8)
X_train, X_test = X[0:size], X[size+1:len(X)]
# grid search per trovare i migliori parametri
#sarima_grid_search(X_train,7)
# plot with the best parameters -> The set of parameters with the minimum AIC is: SARIMA(0, 1, 1)x(0, 1, 1, 7) - AIC:1049.9572865410532
model = sarima_eva(X,(0, 1, 1),(0, 1, 1, 7),7,len(X_train)+1,X_test,X_train)

#proviamo con un test set piu grande
size2 = int(len(X)*0.70)
X_train2, X_test2 = X[0:size2], X[size2+1:len(X)]
# grid search per trovare i migliori parametri
#sarima_grid_search(X_train2,7)
# plot with the best parameters -> The set of parameters with the minimum AIC is: SARIMA(0, 1, 1)x(1, 1, 1, 7) - AIC:869.7733639307531
model = sarima_eva(X,(0, 1, 1),(1, 1, 1, 7),7,len(X_train2)+1,X_test2,X_train2)