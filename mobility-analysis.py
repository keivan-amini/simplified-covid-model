""" 
The aim of this program is to study a mobility dataset in Python, using Pandas library. 
The dataset imported regards the number of motor vehicle detected every day in 2020, for 292 streets in the city of Bologna.
This script has been written in order to find some "mobility parameter" that could fit into a SIR Model code, with the final aim to reconstruct the curve of 2020 hospetalized patients in the city of Bologna due to COVID-19.

@author: Keivan Amini

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Upload the Mobility-Dataset **WARNING: occasionaly could give a KeyError: 'data' since Pandas mess up the name of the first column. FIXME
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

days = list(range(1,367)) # 2020 had 366 days
casalecchio_mobility = ordered_df.loc[ordered_df["Nome via"] == "TANGENZIALE CASALECCHIO-SAN LAZZARO", "Average Daily Mobility"]

assert len(days) == len(casalecchio_mobility), "Something is wrong"

# Let's consider a moving average
raw = casalecchio_mobility.to_numpy()
smoothed_ts = casalecchio_mobility.rolling(7, center=True).mean()

plt.scatter(days,raw/raw.max(), label = 'Raw data', color ='green', marker = "^", s = 4) # normalized to 1
plt.scatter(days,smoothed_ts/smoothed_ts.max(), label = "Moving average based on 7 days", color = "red", marker = "o", s = 4)
plt.axvline(x=70, color='b', ls='--', label='Lockdown') # start of the first italian lockdown
plt.axvline(x=125, color='b', ls='--') # end lockdown
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected normalized')
plt.legend(loc="lower right")
plt.title('Mobility vs Time in Tangenziale Casalecchio - San Lazzaro (BO) during 2020')
plt.show()

# Are the geopoints always the same? Do they change?
trial_df = ordered_df.loc[ordered_df["data"] == "2020-01-01"] # dataset regarding the first day
assert len(pd.unique(df['longitudine'])) == len(pd.unique(trial_df['longitudine'])), "Geopoint are not in the same position everyday"


# Let's reconstruct an array called Average Monthly Mobolity.
# Each element of the 1-D array is associated with a certain geopoint. Let's focus on January.
january_days= ['2020-01-01','2020-01-02','2020-01-03','2020-01-04','2020-01-05'
,'2020-01-06','2020-01-07','2020-01-08','2020-01-09','2020-01-10'
,'2020-01-11','2020-01-12','2020-01-13','2020-01-14','2020-01-15'
,'2020-01-16','2020-01-17','2020-01-18','2020-01-19','2020-01-20'
,'2020-01-21','2020-01-22','2020-01-23','2020-01-24','2020-01-25'
,'2020-01-26','2020-01-27','2020-01-28','2020-01-29','2020-01-30'
,'2020-01-31']

january_df =  ordered_df[ordered_df['data'].isin(january_days)]
simpler_df = january_df[['geopoint','Average Daily Mobility']]

simpler_df = pd.pivot_table(
    simpler_df,
    index='geopoint',
    aggfunc=np.sum,
    fill_value=0,
    )

simpler_df = simpler_df.reset_index()
simpler_df = simpler_df.rename(columns={"Average Daily Mobility": "Total Monthly Mobility"})
simpler_df['Average Monthly Mobility'] = simpler_df['Total Monthly Mobility'] / 31 # number of days


# Let's plot Bologna Map with the size of the points related to Average Monthly Mobility
geopoint_array = simpler_df["geopoint"].str.split(",", n = 1, expand = True)

longitudine_jan = geopoint_array[0].values
longitudine_jan = longitudine_jan.astype(float)

latitudine_jan = geopoint_array[1].values
latitudine_jan = latitudine_jan.astype(float)

Bounding_Box_Jan = ((january_df.longitudine.min(),   january_df.longitudine.max(),      
         january_df.latitudine.min(), january_df.latitudine.max()))

fig, ax = plt.subplots(figsize = (8,7))
s = simpler_df['Average Monthly Mobility'] # size of the circles are related to the Average Monthly Mobility
ax.scatter(latitudine_jan, longitudine_jan, zorder=1, alpha= 0.2, c='b', s = s)
ax.set_title('Geographical position of autoveichle detectors in the city of Bologna, during January 2020')
ax.set_xlim(Bounding_Box_Jan[0],Bounding_Box_Jan[1])
ax.set_ylim(Bounding_Box_Jan[2],Bounding_Box_Jan[3])

ax.imshow(mappa_bologna, zorder=0, extent = Bounding_Box_Jan, aspect= 'equal')
plt.show()

"""
TODO
# fare media totale di tutti i dati sulla mobilità:
# asse x: tempo
# asse y: mobilità totale (somma considerando tutti i luoghi)
"""
