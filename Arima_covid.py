from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

series = read_csv(r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\andamento-nazionale-completo.csv')

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
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# valori di arima non scelti da me; si può andare avanti/modificare impostando giusti valori