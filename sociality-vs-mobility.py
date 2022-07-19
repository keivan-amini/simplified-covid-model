# Sociality vs Mobility code. Data considered: 2020, 2021, 2022.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Social activity rate
m_vals = np.array([1.00, 1.00, 1.00, 0.24, 0.16, 0.17, 0.18, 0.24, 0.24, 0.24, 0.34, 0.24, 0.21, 0.27, 0.25, 0.23, 0.29, 0.26, 0.20, 0.39, 0.29, 0.65, 0.49, 0.35, 0.29, 0.24, 0.19, 0.25, 0.06, 0.08, 0.18, 0.21, 0.18, 0.12, 0.08, 0.06])
m_days = np.array([0, 54, 61, 71, 83, 125, 139, 155, 167, 181, 258, 299, 320, 341, 359, 375, 398, 418, 429, 468, 482, 537, 560, 575, 610, 635, 688, 714, 727, 750, 770, 791, 810, 859, 900, 910])

url_2020 = 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
url_2021 = 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2021_header_mod.csv'
url_2022 = "https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2022_header_mod.csv"

def get_total_mobility(url):

    '''
    Given the link of the dataframe, return an array containing the mobility.

            Parameters:
                    url (str): github url of the dataframe we want to save, i.e. 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
            Return:
                    mobility (np.array): array containing the number of motor veichles detected in that year, WITHOUT normalization or sample rolling.

    '''

    if '2020' in url:
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(url, sep = ';')


    col_list= ['0000 0100', '0100 0200', '0200 0300', "0300 0400" , "0400 0500" , "0500 0600" , "0600 0700" , "0700 0800" , "0800 0900" , "0900 1000" , "1000 1100" , "1100 1200" , "1200 1300" , "1300 1400" , "1400 1500" , "1500 1600" , "1600 1700" , "1700 1800" , "1800 1900" , "1900 2000" , "2000 2100" , "2100 2200" , "2200 2300" , "2300 2400"] 

    ordered_df = df.sort_values(by = 'data')
    ordered_df['Average Daily Mobility'] = ordered_df[col_list].sum(axis=1) / 24

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
    total_mobility = total_mobility[:lastElementIndex]

    return total_mobility

mobility2020 = get_total_mobility(url_2020)
mobility2021 = get_total_mobility(url_2021)
mobility2022 = get_total_mobility(url_2022)
days = np.arange(1, 912) #912 is given by 366 + 365 + 181 (since the 2022 df finishes at 30th June)

total_mobility = pd.concat([mobility2020,mobility2021,mobility2022], ignore_index=True)
smoothed_total = total_mobility.rolling(7, center=True).mean() #rolling mean

# Rolling mean outputs NaN values as first and last element of the array. Let's clean this.
smoothed_total = smoothed_total.to_numpy()
index_of_nan = np.argwhere(np.isnan(smoothed_total))

smoothed_total = np.delete(smoothed_total,index_of_nan) #removing NaN values from the array
days = np.delete(days, index_of_nan) # deleting days associated with NaN values

total_mobility = smoothed_total/smoothed_total.max() #normalization


#Plots
plt.scatter(days, total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
plt.scatter(m_days, m_vals, label = 'Social Activity', color ='deeppink', marker = "+", s = 60)
plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
plt.axvline(x=731, color='darkturquoise', ls='dotted', label = 'End of the year')
plt.title('2020-2022 Mobility vs Time')
plt.xlabel('Number of days from 1 January 2020')
plt.ylabel('Average number of motor vehicle detected normalized')
plt.legend(loc="lower right")
plt.show()


#Shifts
shift_1 = total_mobility[165] - m_vals[7] #165 -> 14 giugno

for index in range(7,21): # dall'ottavo elemento in poi shifto tutti i valori di 0,57
        m_vals[index] += shift_1

plt.scatter(days,total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
plt.scatter(m_days, m_vals, label = 'Social Activity', color ='deeppink', marker = "+", s = 60) # normalized to 1
plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
plt.axvspan(153, 431, facecolor='lightsalmon', alpha=0.2,label = 'Shift = +' + str(round(shift_1,2)))
plt.legend(loc="lower right")
plt.title('Mobility & Sociality (shifted) vs Time')
plt.show()

shift_2 = total_mobility[m_days[25]]- m_vals[25] +0.04
for index in range(24,len(m_vals)):
    m_vals[index] += shift_2

plt.scatter(days,total_mobility, label = "Mobility", color = "green", marker = "o", s = 4)
plt.scatter(m_days, m_vals, label = 'Social Activity', color ='deeppink', marker = "+", s = 60) # normalized to 1
plt.axvline(x=366, color='darkturquoise', ls='dotted', label = 'End of the year')
plt.axvspan(153, 431, facecolor='lightsalmon', alpha=0.2,label = 'Shift = +' + str(round(shift_1,2)))
plt.axvspan(602, 912, facecolor='yellow', alpha=0.2,label = 'Shift = +' + str(round(shift_2,2)))
plt.legend(loc="lower right")
plt.title('Mobility & Sociality (shifted) vs Time')
plt.show()
