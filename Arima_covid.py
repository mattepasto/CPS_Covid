import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.statespace.sarimax as sarima
from sklearn.metrics import mean_squared_error
import itertools

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
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    # fit the model 
    mod = sarima.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    pred = results.get_prediction(start=pred_date, dynamic=False)
    pred_ci = pred.conf_int()
    y_forecasted = pred.predicted_mean
    mse = ((y_forecasted - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = False {}'.format(seasonal_period,round(np.sqrt(mse), 2)))

    ax = plt.plot(y, label='observed')  # modificato da ax = y.plot(label='observed')
    plt.plot(y_forecasted, ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))   # QUI ERRORE
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Data')
    ax.set_ylabel('nuovi_positivi')
    plt.legend()
    plt.show()

    # A better representation of our true predictive power can be obtained using dynamic forecasts. 
    # In this case, we only use information from the time series up to a certain point, 
    # and after that, forecasts are generated using values from previous forecasted time points.
    pred_dynamic = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int()
    y_forecasted_dynamic = pred_dynamic.predicted_mean
    mse_dynamic = ((y_forecasted_dynamic - y_to_test) ** 2).mean()
    print('The Root Mean Squared Error of SARIMA with season_length={} and dynamic = True {}'.format(seasonal_period,round(np.sqrt(mse_dynamic), 2)))

    ax = y.plot(label='observed')
    y_forecasted_dynamic.plot(label='Dynamic Forecast', ax=ax,figsize=(14, 7))
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.2)

    ax.set_xlabel('Data')
    ax.set_ylabel('nuovi_positivi')

    plt.legend()
    plt.show()
    
    return (results)


series = pd.read_csv(r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\dpc-covid19-ita-andamento-nazionale.csv', index_col='data', parse_dates=True)
X = series['nuovi_positivi'].values
# X = X.asfreq('W')

size = int(len(X)*0.8)
X_train, X_test = X[0:size], X[size:len(X)]
# grid search per trovare i migliori parametri
# sarima_grid_search(X,7)
# plot with the best parameters
model = sarima_eva(X,(0, 1, 1),(0, 1, 1, 7),len(X_test),len(X_train+1),X_test)
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