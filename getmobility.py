""" 
The purpose of this code is to implement a function useful to export a np.array mobility in another python code.

"""

import pandas as pd
import numpy as np


def get_mobility(url):

    '''
    Given the url link of the mobility dataframe, return the smoothed and normalized mobility of the dataframe.

            Parameter:
                    url (str): github url of the dataframe we want to save, i.e. 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
            Return:
                    mobility (np.array): array containing the number of motor veichles detected, with a 7-days rolling mean and normalization.

    '''

    if '2020' in url:
        df = pd.read_csv(url)
        days = np.arange(1, 367)
    else:
        df = pd.read_csv(url, sep = ';')
        days = np.arange(1, 365)
        

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


    smoothed_total = total_mobility.rolling(7, center=True).mean() #rolling mean

    # Rolling mean outputs 7 NaN value as first and last element of the array. Let's clean this.
    smoothed_total = smoothed_total.to_numpy()
    index_of_nan = np.argwhere(np.isnan(smoothed_total))
    smoothed_total = np.delete(smoothed_total,index_of_nan) #removing NaN values from the array
    days = np.delete(days, index_of_nan) # deleting days associated with NaN values
    
    mobility = smoothed_total/smoothed_total.max() #normalization

    days = np.append(days, np.inf) # added inf as last element.
    mobility = np.append(mobility, mobility[-1]) #added last element to the mobility array, since mobility and days must be the same size
    number_of_nan = np.count_nonzero(np.isnan(mobility))
    print("Il numero di NaN nell'array mobility è " + str(number_of_nan))

    m = (days, mobility) 
    
    return m


