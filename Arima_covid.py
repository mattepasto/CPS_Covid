import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.tsa.statespace.sarimax as sarima
from sklearn.metrics import mean_squared_error

series = pd.read_csv(r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\dpc-covid19-ita-andamento-nazionale.csv', index_col='data', parse_dates=True)
X = series['nuovi_positivi'].values
# X = X.asfreq('W')

# training
size = int(len(X)*0.8)
X_train, X_test = X[0:size], X[size:len(X)]

# ARIMA model
darlt = sarima.SARIMAX(X_train, order=(1,1,3),trend='t').fit(disp=-1)	# valori scelti da noi, DA MIGLIORARE; 't' = lineare; disp < 0 = non fa visualizzare output
darlf = darlt.forecast(steps=len(X_test))

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