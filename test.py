import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

data = pd.read_csv (r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\dpc-covid19-ita-andamento-nazionale.csv')
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

"""
plt.scatter(giorni, ti, color='black')
plt.grid()
plt.xlabel('Giorni')
plt.ylabel('Terapie intensive')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()

plt.bar(giorni, nuovi_pos)
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()

# data['nuovi_positivi'].plot(kind="hist", title='Nuovi positivi')
# plt.show()

plt.plot(giorni, var_pos)
plt.grid()
plt.xlabel('Giorni')
plt.ylabel('Variazione del totale positivi')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()
"""

# sto provando a fare la single exp smoothing su nuovi positivi (variabile 'nuovi_pos')
"""
index= pd.date_range(start='03/01/2020', end= '04/30/2020')
nuovi_posdata= pd.Series(nuovi_pos,index)

ax= nuovi_posdata.plot()
ax.set_xlabel("Giorni")
ax.set_ylabel("Nuovi positivi")
"""
date_format = [pd.to_datetime(d) for d in giorni]
variable = 'nuovi_positivi'     # è una delle colonne del file csv
fig, ax = plt.subplots(figsize=(12, 5))     # con subplots creo tupla che contiene oggetto figura e assi così ci posso lavorare come entità separate
ax.grid()
ax.scatter(date_format,data[variable])
ax.set(xlabel="Date",ylabel=variable,title=variable)
date_form = DateFormatter("%d-%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.DayLocator(interval = 3))
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
#plt.show()      #04/09: ok fa robe: devo risolvere come mettere indici(probabilmente posso aggiungerlo in un metodo)

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
line_double1, = plt.plot(fcast_double1, marker='o', color='blue')
plt.plot(fit_double2.fittedvalues, marker='o', color='red')
line_double2, = plt.plot(fcast_double2, marker='o', color='red')
plt.plot(fit_double3.fittedvalues, marker='o', color='green')
line_double3, = plt.plot(fcast_double3, marker='o', color='green')
plt.legend([line_double1, line_double2, line_double3], [["alpha=%s, beta=%s", fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend']], ["alpha=%s, beta=%s", fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend']], ["alpha=%s, beta=%s", fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend']]])
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
#plt.show()

# triple exp smoothing: abbiamo notato seasonalità perché nei weekend nuovi_pos minori perché fatti meno tamponi

fit_triple1= ExponentialSmoothing(nuovi_pos, trend= "add", seasonal= "add", seasonal_periods=7).fit()
fcast_triple1 = fit_triple1.forecast(30)#.rename(r'$\alpha=%s, beta=%s$', fit_double1.model.params['smoothing_level'], fit_double1.model.params['smoothing_trend'])
fit_triple2 = ExponentialSmoothing(nuovi_pos, trend= "add", seasonal= "mul", seasonal_periods=7).fit()
fcast_triple2 = fit_triple2.forecast(30)#.rename(r'$\alpha=%s, beta=%s$', fit_double2.model.params['smoothing_level'], fit_double2.model.params['smoothing_trend'])
fit_triple3 = ExponentialSmoothing(nuovi_pos, trend= "mul", seasonal= "add", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple3 = fit_triple3.forecast(30)#.rename(r'$\alpha=%s, beta=%s$', fit_double3.model.params['smoothing_level'], fit_double3.model.params['smoothing_trend'])
fit_triple4 = ExponentialSmoothing(nuovi_pos, trend= "mul", seasonal= "mul", seasonal_periods=7).fit()    # damped non va bene!
fcast_triple4 = fit_triple4.forecast(30)    # PROBABILMENTE LA MIGLIORE PAZZESCO

plt.figure(figsize=(12, 8))
plt.plot(nuovi_pos, marker='o', color='black')
plt.plot(fit_triple1.fittedvalues, marker='o', color='blue')
line_triple1, = plt.plot(fcast_triple1, marker='o', color='blue')
plt.plot(fit_triple2.fittedvalues, marker='o', color='red')
line_triple2, = plt.plot(fcast_triple2, marker='o', color='red')
plt.plot(fit_triple3.fittedvalues, marker='o', color='green')
line_triple3, = plt.plot(fcast_triple3, marker='o', color='green')
plt.plot(fit_triple4.fittedvalues, marker='o', color='brown')
line_triple4, = plt.plot(fcast_triple4, marker='o', color='brown')
plt.legend([line_triple1, line_triple2, line_triple3, line_triple4], [["alpha=%s, beta=%s", fit_triple1.model.params['smoothing_level'], fit_triple1.model.params['smoothing_trend']], ["alpha=%s, beta=%s", fit_triple2.model.params['smoothing_level'], fit_triple2.model.params['smoothing_trend']], ["alpha=%s, beta=%s", fit_triple3.model.params['smoothing_level'], fit_triple3.model.params['smoothing_trend']],["alpha=%s, beta=%s", fit_triple4.model.params['smoothing_level'], fit_triple4.model.params['smoothing_trend']]])
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()
"""
ti_index = pd.date_range(start='01/03/2020', end='30/04/2020')
ti__data = pd.Series(ti, ti_index)
fit1 = SimpleExpSmoothing(ti__data,initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.predict(3).rename(r'$\alpha=0.2$')    # come con altro file, non c'è come metodo forecast ma predict
"""