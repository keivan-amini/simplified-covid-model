""" 
The purpose of this code is to implement a function useful to export a np.array mobility in another python code.

"""

import pandas as pd
import numpy as np
import scipy.interpolate as interpolate

# Social activity rate
m_vals = np.array([1.00, 1.00, 1.00, 0.24, 0.16, 0.17, 0.18, 0.24, 0.24, 0.24, 0.34, 0.24, 0.21, 0.27, 0.25, 0.23, 0.29, 0.26, 0.20, 0.39, 0.29, 0.65, 0.49, 0.35, 0.29, 0.24, 0.19, 0.25, 0.06, 0.08, 0.18, 0.21, 0.18, 0.12, 0.08, 0.06])
m_days = np.array([0, 54, 61, 71, 83, 125, 139, 155, 167, 181, 258, 299, 320, 341, 359, 375, 398, 418, 429, 468, 482, 537, 560, 575, 610, 635, 688, 714, 727, 750, 770, 791, 810, 859, 900, 910])


def get_mobility(url, shift = False):

    '''
    Given the url link of the mobility dataframe, return the smoothed and normalized mobility of the dataframe.

            Parameter:
                    url (str): github url of the dataframe we want to save, i.e. 'https://raw.githubusercontent.com/keivan-amini/simplified-covid-model/main/rilevazione-autoveicoli-tramite-spire-anno-2020_header_mod.csv'
                    shift (bool): if True, shift road mobility so that it will follow social activity trends. The default is False.
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

    index_list = [17, 96, 97, 103, 109, 147, 243, 355]

    def find_outliers(index_list, n=7):
        '''
        Function that find outliers data in this specific scenario.
            Parameters:
                    index_list (list): list containing the first index of the total_mobility array in which wrong measurement start.
                    n (scalar): window size in the moving average. 7 if not specified.
            Return:
                    index_outliers (array): array containing the index related to the outliers in the total_mobility_raw array.
        '''
        index_outliers = [element+(n-1) for element in index_list]
        return index_outliers   

    index_outliers = find_outliers(index_list)

    total_mobility_raw = np.array(total_mobility)
    total_mobility_clean = np.delete(total_mobility_raw,index_outliers)
    days_clean = np.delete(days,index_outliers) # so we can perform interpolation.

# Interpolation
    function = interpolate.interp1d(days_clean, total_mobility_clean, "linear")
    new_x = index_outliers
    interpolated_y = function(new_x) 

# Neatly insert interpolated_y into y
    for element,index in zip(interpolated_y,new_x):
        total_mobility_clean = np.insert(total_mobility_clean, index, element)

# Moving average
    total_mobility_clean = pd.Series(total_mobility_clean) 
    final_mobility_smoothed = total_mobility_clean.rolling(7, center=True).mean()

# Removing NaN
    final_mobility_smoothed = final_mobility_smoothed.to_numpy()
    index_of_nan = np.argwhere(np.isnan(final_mobility_smoothed))
    final_mobility_smoothed = np.delete(final_mobility_smoothed,index_of_nan) 
    days = np.delete(days, index_of_nan) 

# Normalization.
    mobility = final_mobility_smoothed/final_mobility_smoothed.max() 

    if shift == True:
        shift_1 = mobility[165] - m_vals[7] #165 -> 14th June. shift_1 = 0.57
        for index in range(112, len(mobility)): #156 (day from wich shift starts) - 44 (day-zero pandemic) = 112
            mobility[index] -= shift_1
            if mobility[index] < 0:
                mobility[index] = 0.1

    days = np.append(days, np.inf) # added inf as last element.
    mobility = np.append(mobility, mobility[-1]) #added last element to the mobility array, since mobility and days must be the same size

    m = (days, mobility) 
    
    return m