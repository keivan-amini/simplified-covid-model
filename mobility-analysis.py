""" 
The aim of this program is to study a mobility dataset in Python, using Pandas library. 
The dataset imported regards the number of motor vehicle detected every day in 2020, for 292 streets in the city of Bologna.
This script has been written in order to find some "mobility parameter" that could fit into a SIR Model code, with the final aim to reconstruct the curve of 2020 hospetalized patients in the city of Bologna due to COVID-19.

@author: Keivan Amini

"""

import pandas as pd
import matplotlib.pyplot as plt

# Upload the Mobility-Dataset (method inspired by: https://towardsdatascience.com/easy-steps-to-plot-geographic-data-on-a-map-python-11217859a2db)
url = 'https://github.com/keivan-amini/simplified-covid-model/blob/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv?raw=true'
df = pd.read_csv(url,index_col=0)


# Plot a map of the detection spots
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