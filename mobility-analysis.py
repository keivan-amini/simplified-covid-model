""" 
The aim of this program is to study a mobility dataset in Python, using Pandas library. 
The dataset imported regards the number of motor vehicle detected every day in 2020, for 292 streets in the city of Bologna.
This script has been written in order to find some "mobility parameter" that could fit into a SIR Model code, with the final aim to reconstruct the curve of 2020 hospetalized patients in the city of Bologna due to COVID-19.

@author: Keivan Amini

"""

import pandas as pd
import matplotlib.pyplot as plt


# Upload the Mobility-Dataset 
url = 'https://github.com/keivan-amini/simplified-covid-model/blob/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv?raw=true'
df = pd.read_csv(url,index_col=0)


# Plot a map of the detection spots (method inspired by: https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db)
Bounding_Box = ((df.longitudine.min(),   df.longitudine.max(),      
         df.latitudine.min(), df.latitudine.max()))

mappa_bologna = plt.imread("https://github.com/keivan-amini/simplified-covid-model/blob/main/mappa1080.png?raw=true")

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(pd.unique(df['longitudine']), pd.unique(df['latitudine']), zorder=1, alpha= 0.2, c='b', s=10)
ax.set_title('Geographical position of autoveichle detectors in the city of Bologna')
ax.set_xlim(Bounding_Box[0],Bounding_Box[1])
ax.set_ylim(Bounding_Box[2],Bounding_Box[3])

ax.imshow(mappa_bologna, zorder=0, extent = Bounding_Box, aspect= 'equal')
plt.show()


# Let's consider a mean of the mobility for every day, and then compute a plot example of mobility vs time
ordered_df = df.sort_values(by = 'data') 
col_list= ['0000 0100', '0100 0200', '0200 0300', "0300 0400" , "0400 0500" , "0500 0600" , "0600 0700" , "0700 0800" , "0800 0900" , "0900 1000" , "1000 1100" , "1100 1200" , "1200 1300" , "1300 1400" , "1400 1500" , "1500 1600" , "1600 1700" , "1700 1800" , "1800 1900" , "1900 2000" , "2000 2100" , "2100 2200" , "2200 2300" , "2300 2400"] 
ordered_df['Average Daily Mobility'] = ordered_df[col_list].sum(axis=1) / 24

days = list(range(1,367))
casalecchio_mobility = ordered_df.loc[ordered_df["Nome via"] == "TANGENZIALE CASALECCHIO-SAN LAZZARO", "Average Daily Mobility"]

assert len(days) == len(casalecchio_mobility), "Something is wrong"

plt.plot(days, casalecchio_mobility,label = 'Tangenziale Casalecchio - San Lazzaro', color='green', linewidth = 1) 
plt.axvline(x=70, color='b', ls='--', label='31 marzo 2020') # start of the first italian lockdown
plt.axvline(x=125, color='b', ls='--', label='4 maggio 2020') # end lockdown
plt.xlabel('Days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected')
plt.title('Mobility vs Time in Tangenziale Casalecchio - San Lazzaro (BO) during 2020')
plt.show()
