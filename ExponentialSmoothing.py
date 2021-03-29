import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

data = pd.read_csv (r'.\datasets\dpc-covid19-ita-andamento-nazionale.csv')
#print(data)
giorni = data['data']
ti = data['terapia_intensiva']
ricoverati = data['ricoverati_con_sintomi']
isol_dom = data['isolamento_domiciliare']
tot_pos = data['totale_positivi']
var_pos = data['variazione_totale_positivi']
nuovi_pos = data['nuovi_positivi']
dimessi = data['dimessi_guariti']
decessi = data['deceduti']
tot_casi = data['totale_casi']
tamponi = data['tamponi']

date_format = [pd.to_datetime(d) for d in giorni]
variable = 'nuovi_positivi'     # è una delle colonne del file csv
fig, ax = plt.subplots(figsize=(12, 5))     # con subplots creo tupla che contiene oggetto figura e assi così ci posso lavorare come entità separate
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
plt.title('nuovi_positivi')
#plt.show()

fit1= SimpleExpSmoothing(nuovi_pos, initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.forecast(3).rename(r'$\alpha=0.2$')    #0.2 è valore basso -> faccio pesare di più storia pregressa; per capire significato paramentro alpha vedi: https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html
fit2 = SimpleExpSmoothing(nuovi_pos, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
fcast2 = fit2.forecast(3).rename(r'$\alpha=0.6$')
fit3 = SimpleExpSmoothing(nuovi_pos, initialization_method="estimated").fit()   # qui alpha lo facciamo decidere a statsmodel: troviamo valore ottimizzato ed è approccio migliore
fcast3 = fit3.forecast(3).rename(r'$\alpha estimated=%s$'%fit3.model.params['smoothing_level'])


plt.figure(figsize=(12, 8))
plt.plot(nuovi_pos, marker='o', color='black')
plt.plot(fit1.fittedvalues, marker='o', color='blue')
line1, = plt.plot(fcast1, marker='o', color='blue')
plt.plot(fit2.fittedvalues, marker='o', color='red')
line2, = plt.plot(fcast2, marker='o', color='red')
plt.plot(fit3.fittedvalues, marker='o', color='green')
line3, = plt.plot(fcast3, marker='o', color='green')
plt.legend([line1, line2, line3], [fcast1.name, fcast2.name, fcast3.name])
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.title('SimpleExpSmoothing')
#plt.show()

# double exp smoothing

fit_double1= ExponentialSmoothing(nuovi_pos, trend= "add").fit()
fcast_double1 = fit_double1.forecast(15)#.rename(r'$\alpha=%s, beta=%s$', fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend'])
fit_double2 = ExponentialSmoothing(nuovi_pos, trend= "mul").fit()
fcast_double2 = fit_double2.forecast(15)#.rename(r'$\alpha=%s, beta=%s$', fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend'])
fit_double3 = ExponentialSmoothing(nuovi_pos, trend="add", damped_trend= True).fit()    # damped non va bene!
fcast_double3 = fit_double3.forecast(15)#.rename(r'$\alpha=%s, beta=%s$', fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend'])

plt.figure(figsize=(12, 8))
plt.plot(nuovi_pos, marker='o', color='black')
plt.plot(fit_double1.fittedvalues, marker='o', color='blue')
line_double1, = plt.plot(fcast_double1, marker='o', color='blue',label="alpha = %.2f, beta = %.2f" %(fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend']))
plt.plot(fit_double2.fittedvalues, marker='o', color='red')
line_double2, = plt.plot(fcast_double2, marker='o', color='red',label="alpha = %.2f, beta = %.2f" %(fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend']))
plt.plot(fit_double3.fittedvalues, marker='o', color='green')
line_double3, = plt.plot(fcast_double3, marker='o', color='green',label="alpha = %.2f, beta = %.2f" %(fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend']))
plt.legend(handles=[line_double1, line_double2, line_double3])
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.title('DoubleExpSmoothing')
#plt.show()

# triple exp smoothing: abbiamo notato stagionalità perché nei weekend nuovi_pos minori perché fatti meno tamponi

size = int(len(nuovi_pos)*0.8)
nuovi_pos_train=nuovi_pos[0:size]
nuovi_pos_test=nuovi_pos[size:len(nuovi_pos)]

fit_triple1= ExponentialSmoothing(nuovi_pos_train, trend= "add", seasonal= "add", seasonal_periods=7).fit()
fcast_triple1 = fit_triple1.forecast(len(nuovi_pos)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend'])
fit_triple2 = ExponentialSmoothing(nuovi_pos_train, trend= "add", seasonal= "mul", seasonal_periods=7).fit()
fcast_triple2 = fit_triple2.forecast(len(nuovi_pos)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend'])
fit_triple3 = ExponentialSmoothing(nuovi_pos_train, trend= "mul", seasonal= "add", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple3 = fit_triple3.forecast(len(nuovi_pos)-size)#.rename(r'$\alpha=%s, beta=%s$', fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend'])
fit_triple4 = ExponentialSmoothing(nuovi_pos_train, trend= "mul", seasonal= "mul", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple4 = fit_triple4.forecast(len(nuovi_pos)-size)    # PROBABILMENTE LA MIGLIORE

mse1 = ((fcast_triple1 - nuovi_pos_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = add and seasonal = add is {}'.format(round(np.sqrt(mse1), 2)))
mse2 = ((fcast_triple2 - nuovi_pos_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = add and seasonal = mul is {}'.format(round(np.sqrt(mse2), 2)))
mse3 = ((fcast_triple3 - nuovi_pos_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = mul and seasonal = add is {}'.format(round(np.sqrt(mse3), 2)))
mse4 = ((fcast_triple4 - nuovi_pos_test) ** 2).mean()
print('The Root Mean Squared Error of TripleExpSmoothing with trend = mul and seasonal = mul is {}'.format(round(np.sqrt(mse4), 2)))

fig,axs=plt.subplots(2,2)
axs[0,0].plot(nuovi_pos, marker='o', color='black')
axs[0,0].plot(fit_triple1.fittedvalues, marker='o', color='blue')
axs[0,0].plot(fcast_triple1, marker='o', color='cyan')
plt.sca(axs[0, 0])
plt.xticks([0,10,20,30,40,50,60],["1 Mar", "11 Mar", "21 Mar", "31 Mar", "10 Apr", "20 Apr", "30 Apr"])
axs[0,0].set_title("add add")

axs[0,1].plot(nuovi_pos, marker='o', color='black')
axs[0,1].plot(fit_triple2.fittedvalues, marker='o', color='red')
axs[0,1].plot(fcast_triple2, marker='o', color='cyan')
plt.sca(axs[0, 1])
plt.xticks([0,10,20,30,40,50,60],["1 Mar", "11 Mar", "21 Mar", "31 Mar", "10 Apr", "20 Apr", "30 Apr"])
axs[0,1].set_title("add mul")

axs[1,0].plot(nuovi_pos, marker='o', color='black')
axs[1,0].plot(fit_triple3.fittedvalues, marker='o', color='green')
axs[1,0].plot(fcast_triple3, marker='o', color='cyan')
plt.sca(axs[1, 0])
plt.xticks([0,10,20,30,40,50,60],["1 Mar", "11 Mar", "21 Mar", "31 Mar", "10 Apr", "20 Apr", "30 Apr"])
axs[1,0].set_title("mul add")

axs[1,1].plot(nuovi_pos, marker='o', color='black')
axs[1,1].plot(fit_triple4.fittedvalues, marker='o', color='brown')
axs[1,1].plot(fcast_triple4, marker='o', color='cyan')
plt.sca(axs[1, 1])
plt.xticks([0,10,20,30,40,50,60],["1 Mar", "11 Mar", "21 Mar", "31 Mar", "10 Apr", "20 Apr", "30 Apr"])
axs[1,1].set_title("mul mul")
fig.suptitle('TripleExpSmoothing')
plt.show()