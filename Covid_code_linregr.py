# SOLO TREND NAZIONALE PER ORA

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from sklearn import linear_model
from sklearn.metrics import max_error
import math

# importazione dati
data = pd.read_csv(r'.\datasets\dpc-covid19-ita-andamento-nazionale.csv')
# print (data.columns)

data['diff_deceduti'] = data['deceduti'].diff()     # differenza con giorno prima (metodo diff non ha parametri quindi per default passo è 1)
data['diff_tamponi'] = data['tamponi'].diff()
dates = data['data']
date_format = [pd.to_datetime(d) for d in dates]

variable = 'nuovi_positivi'     # è una delle colonne del file csv
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Nuovi positivi')
plt.show()  # stampo semplicemente nuovi pos

# prima correzione ad anomalie dovute a weekend, "levigando la curva"
rolling_average_days = 7
data['nuovi_positivi_moving'] = data['nuovi_positivi'].rolling(window=rolling_average_days).mean()  # rolling è di pandas e lavora con dataframe. Dò stesso peso a tutti i dati
variable = 'nuovi_positivi_moving'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Nuovi positivi levigato')
plt.show()

# ma i tamponi? notiamo che numero contagiati è strettamente collegato a numero tamponi fatti
data['diff_tamponi_moving'] = data['tamponi'].diff().rolling(window=rolling_average_days).mean()
variable = 'diff_tamponi_moving'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Numero tamponi levigato')
plt.show()

# allora vediamo la percentuale di positivi in base a numero di test
data['perc_positive'] = ((data['nuovi_positivi'])/(data['diff_tamponi'])*100)
variable = 'perc_positive'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Percentuale positivi (n° pos / n° tamponi)')
plt.show()

# allora vediamo la percentuale di positivi in base a numero di test
data['perc_positive_moving'] = ((data['nuovi_positivi_moving'])/(data['diff_tamponi_moving'])*100)
variable = 'perc_positive_moving'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Percentuale positivi levigato(n° pos / n° tamponi)')
plt.show()      # percentuale risolve problema delle fluttazioni dei weekend e pesa il numero dei test

# vediamo grafico terapie intensive e differenza deceduti: il secondo tra questi è quello più 'grezzo'
variable = 'terapia_intensiva'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Terapie intensive occupate')
plt.show()

variable = 'diff_deceduti'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('Deceduti giornalieri')
plt.show()

# mettiamo insieme le terapie intensive con i morti: l'insieme di questi dati ci dà la certezza di non avere errori di acquisizione (ovvero è meno proponso a una forte fluttuazione come il numero di tamponi o contagi)
data['gravi_deceduti'] = data['diff_deceduti'] + data['terapia_intensiva']
variable = 'gravi_deceduti'
fig, ax = plt.subplots(figsize=(12, 5))
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
ax.axvline(datetime(2020, 4, 1), c="green", zorder=0)   # picco
plt.title('Numero persone gravi + deceduti')
plt.show()

# ora uso regressione lineare su terapie + morti
# prepare the lists for the model
X = date_format
y = data['gravi_deceduti'].tolist()[1:]
# per la modellazione, coverto date in numeri incrementali
starting_date = 37  # il primo aprile è il 37° giorno della serie
day_numbers = []
for i in range(1, len(X)):
    day_numbers.append([i])
X = day_numbers
# # let's train our model only with data after the peak
X = X[starting_date:]
y = y[starting_date:]
# Instantiate Linear Regression
linear_regr = linear_model.LinearRegression()
# Train the model using the training sets
linear_regr.fit(X, y)
print ("Linear Regression Model Score: %s" % (linear_regr.score(X, y)))

# Predict future trend
y_pred = linear_regr.predict(X)
error = max_error(y, y_pred)
X_test = []
future_days = 60    # predico prossimo mese in questo caso maggio
for i in range(starting_date, starting_date + future_days):
    X_test.append([i])
y_pred_linear = linear_regr.predict(X_test)
y_pred_max = [] # è predizone + errore
y_pred_min = [] # è predizione - errore
for i in range(0, len(y_pred_linear)):
    y_pred_max.append(y_pred_linear[i] + error)
    y_pred_min.append(y_pred_linear[i] - error)

# convert date of the epidemic peak into datetime format
date_zero = datetime.strptime(data['data'][starting_date], '%Y-%m-%dT%H:%M:%S')
# creating x_ticks for making the plot more appealing
date_prev = []
x_ticks = []
step = 3    # vedo progressi giorno dopo giorno
data_curr = date_zero
x_current = starting_date
n = int(future_days / step)
for i in range(0, n):
    date_prev.append(str(data_curr.day) + "/" + str(data_curr.month))
    x_ticks.append(x_current)
    data_curr = data_curr + timedelta(days=step)
    x_current = x_current + step

# plot known data
plt.grid()
plt.scatter(X, y)
# plot linear regression prediction
plt.plot(X_test, y_pred_linear, color='green', linewidth=2)
# plot maximum error
plt.plot(X_test, y_pred_max, color='red', linewidth=1, linestyle='dashed')
#plot minimum error
plt.plot(X_test, y_pred_min, color='red', linewidth=1, linestyle='dashed')
plt.xlabel('Days')
plt.xlim(starting_date, starting_date + future_days)
plt.xticks(x_ticks, date_prev)
plt.ylabel('gravi_deceduti')
plt.yscale("log")   # log o linear; più bello log ma entrambi vanno a zero il 18 Maggio
plt.title('Predizione decessi e casi gravi')
plt.show()