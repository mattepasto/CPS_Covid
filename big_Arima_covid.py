import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.statespace.sarimax as sarima
from sklearn.metrics import mean_squared_error
import itertools

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
    asse_x2=np.arange(len(y_train),len(y),1)
    ax.plot(asse_x,y_train, label='X_train')
    ax.plot(asse_x2,y_to_test, label='X_test')
    ax.plot(asse_x2,y_forecasted, label='y_forecasted')
    ax.plot(asse_x2,y_forecasted_dynamic, label='y_forecasted_dynamic')
    plt.legend(loc='upper left')
    plt.title('SARIMAX')
    plt.ylabel('nuovi_positivi')
    plt.xlabel('Data')
    plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95],
    ["24 Feb","1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr",
    "5 Mag","10 Mag","15 Mag","20 Mag","25 Mag", "30 Mag"])
    plt.show()
    return (results)
    

series = pd.read_csv(r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\andamento-nazionale-completo.csv', index_col='data', parse_dates=True)
X = series['nuovi_positivi'].values
# X = X.asfreq('W')

size = int(len(X)*0.8)
X_train, X_test = X[0:size], X[size:len(X)]
# grid search per trovare i migliori parametri
#sarima_grid_search(X_train,7)
# plot with the best parameters -> The set of parameters with the minimum AIC is: SARIMA(0, 1, 1)x(1, 1, 1, 7) - AIC:933.6363644269596
model = sarima_eva(X,(0, 1, 1),(1, 1, 1, 7),7,len(X_train+1),X_test,X_train)

#proviamo con un test set piu grande
size2 = int(len(X)*0.70)
X_train2, X_test2 = X[0:size2], X[size2:len(X)]
# grid search per trovare i migliori parametri
#sarima_grid_search(X_train2,7)
# plot with the best parameters -> The set of parameters with the minimum AIC is: SARIMA(0, 1, 1)x(1, 1, 1, 7) - AIC:933.6363644269596
model = sarima_eva(X,(0, 1, 1),(1, 1, 1, 7),7,len(X_train2+1),X_test2,X_train2)
'''
# training
size = int(len(X)*0.8)
X_train, X_test = X[0:size], X[size:len(X)]

# ARIMA model
darlt = sarima.SARIMAX(X_train, order=(1,1,3),trend='t').fit(disp=-1)	# valori scelti da noi, DA MIGLIORARE; 't' = lineare; disp < 0 = non fa visualizzare output
darlf = darlt.forecast(steps=len(X_test))

# Compute root mean square forecasting error
true = series.reindex(darlf.index)
error = true - darlf

# Print out the results
print(pd.concat([true.rename('true'),
                 darlf.rename('forecast'),
                 error.rename('error')], axis=1))
# plot
fig, ax = plt.subplots()
asse_x=np.arange(0,size,1)
asse_x2=np.arange(size,len(X),1)
ax.plot(asse_x,X_train, label='X_train')
ax.plot(asse_x2,X_test, label='X_test')
ax.plot(asse_x2,darlf, label='darlf')
plt.legend(loc='upper left')
plt.title('SARIMAX')
plt.ylabel('nuovi_positivi')
plt.xlabel('Data')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()
'''


'''
---------------------------------------------------------------------
# divido dataset in due: il primo è il training set che uso per il modello ARIMA; il secondo è il dataset di test che supponiamo non disponibile

X = series['nuovi_positivi'].values
size = int(len(X)*0.66)     # prendo 2/3 di intero dataset
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]   #key error??
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

# valori di arima non scelti da me; si può andare avanti/modificare impostando giusti valori
'''