""" 
The aim of this program is to study a mobility dataset in Python, using Pandas library. 
The dataset imported regards the number of motor vehicle detected every day in 2020, for 292 streets in the city of Bologna.
This script has been written in order to find some "mobility parameter" that could fit into a SIR Model code, with the final aim to reconstruct the curve of 2020 hospetalized patients in the city of Bologna due to COVID-19.

@author: Keivan Amini

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


# Upload the Mobility-Dataset **WARNING: occasionaly could give a KeyError: 'data' since Pandas mess up the name of the first column. FIXME
#url = 'https://github.com/keivan-amini/simplified-covid-model/blob/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv?raw=true'
url = 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
#df = pd.read_csv(url,index_col=0)
df = pd.read_csv(url)


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
plt.axvline(x=70, color='b', ls='--', label='Lockdown') # start of the first italian lockdown, 9th march
plt.axvline(x=125, color='b', ls='--') # end lockdown, 18th may
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected normalized')
plt.legend(loc="lower right")
plt.title('Mobility vs Time in Tangenziale Casalecchio - San Lazzaro (BO) during 2020')
plt.show()

# Are the geopoints always the same? Do they change?
trial_df = ordered_df.loc[ordered_df["data"] == "2020-01-01"] # dataset regarding the first day
# assert len(pd.unique(df['longitudine'])) == len(pd.unique(trial_df['longitudine'])), "Geopoint are not in the same position everyday"


# Let's reconstruct an array called Average Monthly Mobility.
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
s = simpler_df['Average Monthly Mobility'] #size of the circles are related to the Average Monthly Mobility
scatter = ax.scatter(latitudine_jan, longitudine_jan, zorder=1, alpha= 0.2, c='b', s = s)
ax.set_title('Geographical position of autoveichle detectors in the city of Bologna, during January 2020')
ax.set_xlim(Bounding_Box[0],Bounding_Box[1])
ax.set_ylim(Bounding_Box[2],Bounding_Box[3])
ax.legend(*scatter.legend_elements("sizes", num=5, color = "b"), loc='lower left',labelspacing = 3, borderpad=2, handletextpad =1.5)
ax.imshow(mappa_bologna, zorder=0, extent = Bounding_Box, aspect= 'equal')
plt.show()


# Let's consider ALL the mobility in the city. 
global_df = pd.pivot_table(
    ordered_df,
    index='data',
    columns='Nome via',
    values='Average Daily Mobility',
    aggfunc=np.sum,
    fill_value=0,
    margins=True,
)

total_mobility = global_df['All']
lastElementIndex = len(total_mobility)-1
total_mobility = total_mobility[:lastElementIndex] #removing the last element from the array since it corresponds to the sum of the total mobility

smoothed_total = total_mobility.rolling(7, center=True).mean()

plt.scatter(days,total_mobility/total_mobility.max(), label = 'Raw data', color ='green', marker = "^", s = 4) # normalized to 1
plt.scatter(days,smoothed_total/smoothed_total.max(), label = "Moving average based on 7 days", color = "red", marker = "o", s = 4)
plt.axvline(x=70, color='b', ls='--', label='1st Lockdown') # start of the first italian lockdown, 9th march
plt.axvline(x=125, color='b', ls='--') # end lockdown, 18th may
plt.axvline(x=286, color='c', ls='--', label='Start of the 2nd wave') # start of the second wave, 13th october, mandatory mask
#plt.axvline(x=291, color='c', ls='--') # 18th october: chiusura scuola e università
plt.axvline(x=297, color='y', ls='--', label='Closure of activities') # 24th october: chiusura attività
plt.axvline(x=319, color='m', ls='--', label = 'Night curfew and zone colours') # 3th november, curfew, introduzione dei colori
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected normalized')
plt.legend(loc="lower right")
plt.title('Total Mobility vs Time in Bologna during 2020')
plt.show()

# comparison
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
ax1.set_title('Mobility in Tangenziale Casalecchio - San Lazzaro (BO)')
ax1.scatter(days, smoothed_ts, s=5)
ax2.set_title('Total Mobility in Bologna')
ax2.scatter(days, smoothed_total, s=5)
fig.tight_layout()
plt.show()

#--------------------------------------------
#--------------------------------------------

# Urban vs Suburban Mobility & Time slots Mobility

borders = [(44.505796, 11.339432), #coordinate geografiche dei confini del centro di bologna
(44.499409, 11.326833),
(44.490256, 11.329385),
(44.486264, 11.339472),
(44.484313, 11.356326),
(44.485729, 11.358204),
(44.501035, 11.356401),
(44.504251, 11.348351)]
polygon = Polygon(borders)

geopoint = pd.unique(ordered_df['geopoint'])
bool = [0] * len(geopoint) #vettore che dovrà contenere True se la coordinata geografica è all'interno del centro di Bologna e False viceversa

for coordinate, index in zip(geopoint, range(len(geopoint))):
        tupla = tuple(map(float, coordinate.split(',')))
        punto = Point(tupla)
        bool[index] = polygon.contains(punto)
        
series = pd.Series(bool)

frame = { 'Geopoint': geopoint, 'Boolean': series }
central_df = pd.DataFrame(frame) #creazione del dataframe
urban_df = central_df.query("Boolean==True")
suburban_df = central_df.query("Boolean==False")

#We have now splitted the geopoint. Now the idea is to analyze the mobility.
urban_geopoint_array = urban_df["Geopoint"].str.split(",", n = 1, expand = True)
suburban_geopoint_array = suburban_df["Geopoint"].str.split(",", n = 1, expand = True)

longitudine_urban = urban_geopoint_array[0].values
longitudine_urban = longitudine_urban.astype(float)
latitudine_urban = urban_geopoint_array[1].values
latitudine_urban = latitudine_urban.astype(float)

longitudine_suburban = suburban_geopoint_array[0].values
longitudine_suburban = longitudine_suburban.astype(float)
latitudine_suburban = suburban_geopoint_array[1].values
latitudine_suburban = latitudine_suburban.astype(float)

#draw a polygon that rapresents the border of the centre
borders.append(borders[0]) #repeat the first point to create a 'closed loop'
xs, ys = zip(*borders) #create lists of x and y values

plt.figure()
plt.plot(ys,xs, color= "r", linewidth="3") 
plt.scatter(latitudine_urban,longitudine_urban)
plt.title('Urban detector coordinates')
plt.xlabel('Longitudine')
plt.ylabel('Latitudine')
plt.xlim([df.longitudine.min(), df.longitudine.max()])
plt.ylim([df.latitudine.min(), df.latitudine.max()])
plt.show()

plt.figure()
plt.plot(ys,xs, color= "r", linewidth="3") 
plt.scatter(latitudine_suburban,longitudine_suburban)
plt.title('Suburban detector coordinates')
plt.xlabel('Longitudine')
plt.ylabel('Latitudine')
plt.xlim([df.longitudine.min(), df.longitudine.max()])
plt.ylim([df.latitudine.min(), df.latitudine.max()])

plt.show()

# now lets plot just the urban mobility. In order to do that i have to cut the dataset just for some geopoints.
#---------------prova
geopoint_urbani = urban_df['Geopoint']
geopoint_suburbani = suburban_df['Geopoint']

list_urban_geopoint = geopoint_urbani.tolist()
list_suburban_geopoint = geopoint_suburbani.tolist()

suburban_df = ordered_df[~ordered_df['geopoint'].isin(list_urban_geopoint)]
urban_df = ordered_df[~ordered_df['geopoint'].isin(list_suburban_geopoint)]

urban_df = pd.pivot_table(
    urban_df,
    index='data',
    columns= 'geopoint',
    values='Average Daily Mobility',
    aggfunc=np.sum,
    fill_value=0,
    margins=True,
)
suburban_df = pd.pivot_table(
    suburban_df,
    index='data',
    columns= 'geopoint',
    values='Average Daily Mobility',
    aggfunc=np.sum,
    fill_value=0,
    margins=True,
)

urban_mobility = urban_df['All']
suburban_mobility = suburban_df['All']

lastElementIndex = len(urban_mobility)-1

urban_mobility = urban_mobility[:lastElementIndex]
suburban_mobility = suburban_mobility[:lastElementIndex]

smoothed_urban = urban_mobility.rolling(7, center=True).mean()
smoothed_suburban = suburban_mobility.rolling(7, center=True).mean()

#plt.scatter(days,urban_mobility/urban_mobility.max(), label = 'Urban Mobility', color ='green', marker = "^", s = 4) # normalized to 1
plt.scatter(days,smoothed_suburban/smoothed_suburban.max(), label = "Suburban Mobility", color = "orangered", marker = "o", s = 4)
plt.scatter(days,smoothed_urban/smoothed_urban.max(), label = "Urban Mobility", color = "limegreen", marker = "o", s = 4)
plt.axvline(x=70, color='b', ls='--', label='Lockdown') # start of the first italian lockdown
plt.axvline(x=125, color='b', ls='--') # end lockdown
plt.axvline(x=286, color='c', ls='--', label='Start of the 2nd wave') # start of the second wave, 13th october, mandatory mask
#plt.axvline(x=291, color='c', ls='--') # 18th october: chiusura scuola e università
plt.axvline(x=297, color='y', ls='--', label='Closure of activities') # 24th october: chiusura attività
plt.axvline(x=319, color='m', ls='--', label = 'Night curfew and zone colours') # 3th november, curfew, introduzione dei colori
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected normalized')
plt.legend(loc="lower right")
plt.title('Urban and Suburban Mobility vs Time in Bologna during 2020')
plt.show()

##not normalized
plt.scatter(days,smoothed_suburban, label = "Suburban Mobility", color = "orangered", marker = "o", s = 4)
plt.scatter(days,smoothed_urban, label = "Urban Mobility", color = "limegreen", marker = "o", s = 4)
plt.axvline(x=70, color='b', ls='--', label='Lockdown') # start of the first italian lockdown
plt.axvline(x=125, color='b', ls='--') # end lockdown
plt.axvline(x=286, color='c', ls='--', label='Start of the 2nd wave') # start of the second wave, 13th october, mandatory mask
#plt.axvline(x=291, color='c', ls='--') # 18th october: chiusura scuola e università
plt.axvline(x=297, color='y', ls='--', label='Closure of activities') # 24th october: chiusura attività
plt.axvline(x=319, color='m', ls='--', label = 'Night curfew and zone colours') # 3th november, curfew, introduzione dei colori
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected')
plt.legend(loc="lower right")
plt.title('Urban and Suburban Mobility (NOT NORMALIZED) vs Time in Bologna during 2020')
plt.show()

# Graphical comparison between coordinates and mobility.
fig, axs = plt.subplots(2, 2)

axs[0, 0].scatter(latitudine_urban, longitudine_urban, s = 1.2)
axs[0, 0].plot(ys,xs, color= "r", linewidth="3")
axs[0,0].set_xlim([df.longitudine.min(), df.longitudine.max()])
axs[0,0].set_ylim([df.latitudine.min(), df.latitudine.max()])
axs[0, 0].set_title('Urban detector coordinates')

axs[1, 0].scatter(latitudine_suburban, longitudine_suburban, s = 1.2, color = "green")
axs[1, 0].plot(ys,xs, color= "r", linewidth="3")
axs[1, 0].set_title('Suburban detector coordinates')

axs[0, 1].scatter(days,smoothed_urban, s = 1.2)
axs[0, 1].set_title('Urban mobility vs Time')

axs[1, 1].scatter(days, smoothed_suburban, s = 1.2, color = "green")
axs[1, 1].set_title('Suburban mobility vs Time')

plt.show()

# Now lets clean the dataset in order to compare morning, afternoon and night.

del ordered_df['codice spira']
del ordered_df['Livello']
del ordered_df['codimpsem']
del ordered_df['angolo']
del ordered_df['geopoint']
del ordered_df['Giorno della settimana']
del ordered_df['Average Daily Mobility']
del ordered_df['latitudine']
del ordered_df['longitudine']
del ordered_df['codice']
del ordered_df['ordinanza']
del ordered_df['id_uni']
del ordered_df['Nodo da']
del ordered_df['direzione']
del ordered_df['stato']
del ordered_df['codice arco']
del ordered_df['codice via']
del ordered_df['Nome via']
del ordered_df['tipologia']
del ordered_df['Nodo a']

hourly_df = ordered_df.groupby('data').sum()

morning = ['0600 0700', '0700 0800', '0800 0900', '0900 1000']
afternoon = ['1600 1700', '1700 1800', '1800 1900', '1900 2000']
night = ['2000 2100', '2100 2200', '2200 2300', '2300 2400']

morning_mobility = hourly_df[morning].sum(axis=1)
afternoon_mobility = hourly_df[afternoon].sum(axis=1)
night_mobility = hourly_df[night].sum(axis=1)

smoothed_morning = morning_mobility.rolling(7, center=True).mean()
smoothed_afternoon = afternoon_mobility.rolling(7, center=True).mean()
smoothed_night = night_mobility.rolling(7, center=True).mean()

with plt.style.context('Solarize_Light2'):
        plt.grid(False)
        #plt.scatter(days,morning_mobility/morning_mobility.max(), label = 'Raw data', color ='green', marker = "^", s = 4) # normalized to 1
        plt.scatter(days,smoothed_morning/smoothed_morning.max(), label = "Morning Mobility 6-10 AM", color = "red", marker = "o", s = 4)
        plt.scatter(days,smoothed_afternoon/smoothed_afternoon.max(), label = "Afternoon Mobility 4-8 PM", color = "g", marker = "o", s = 4)
        plt.scatter(days,smoothed_night/smoothed_night.max(), label = "Night Mobility 9-00 PM", color = "k", marker = "o", s = 4)
        plt.axvline(x=70, color='b', ls='--', label='1st Lockdown') # start of the first italian lockdown, 9th march
        plt.axvline(x=125, color='b', ls='--') # end lockdown, 18th may
        plt.axvline(x=286, color='c', ls='--', label='Start of the 2nd wave') # start of the second wave, 13th october, mandatory mask
#plt.axvline(x=291, color='c', ls='--') # 18th october: chiusura scuola e università
        plt.axvline(x=297, color='y', ls='--', label='Closure of activities') # 24th october: chiusura attività
        plt.axvline(x=319, color='m', ls='--', label = 'Night curfew and zone colours') # 3th november, curfew, introduzione dei colori
        plt.xlabel('Number of days from 1 January 2020', color = 'k')
        plt.ylabel('Average number of motor vehicle detected normalized', color = 'k')
        plt.legend(loc="lower right")
        plt.title('Time Slots Mobility vs Time in Bologna during 2020')
        
plt.show()
