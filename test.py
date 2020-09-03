import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt  # serve per single exp smoothing

data = pd.read_csv (r'C:\Users\matte\Documents\Covid_Machine_Learning\datasets\dpc-covid19-ita-andamento-nazionale.csv')
#print(data)
giorno = data['data']
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
plt.scatter(giorno, ti, color='black')
plt.grid()
plt.xlabel('Giorni')
plt.ylabel('Terapie intensive')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()

plt.bar(giorno, nuovi_pos)
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()

# data['nuovi_positivi'].plot(kind="hist", title='Nuovi positivi')
# plt.show()

plt.plot(giorno, var_pos)
plt.grid()
plt.xlabel('Giorni')
plt.ylabel('Variazione del totale positivi')
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60],
 ["1 Mar", "6 Mar", "11 Mar", "16 Mar", "21 Mar", "26 Mar", "31 Mar", "5 Apr", "10 Apr", "15 Apr", "20 Apr", "25 Apr", "30 Apr"])
plt.show()
"""

# sto provando a fare la single exp smoothing

ti_index = pd.date_range(start='01/03/2020', end='30/04/2020')
ti__data = pd.Series(ti, ti_index)
fit1 = SimpleExpSmoothing(ti__data,initialization_method="heuristic").fit(smoothing_level=0.2,optimized=False)
fcast1 = fit1.predict(3).rename(r'$\alpha=0.2$')    # come con altro file, non c'è come metodo forecast ma predict