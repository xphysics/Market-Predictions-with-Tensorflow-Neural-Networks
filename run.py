import pandas as pd
import numpy as np
from neuralnet import *


mod = TimeSeries()

df = mod.dataframe


def first_available_at(df):
    ''' PURPOSE: identify the first available data's stamp
                for all financial variables given
        RETURN: sorted series of variables and their stamps
    '''
    variables = []
    stamps = []
    for ind in df.columns:
        variables.append(ind)
        stamps.append(df[ind].dropna().index[0])

    result = pd.Series(data = stamps, index = variables)
    
    return result.sort_values()


first = first_available_at(df)


def separate_dataset(df):
    first = first_available_at(df)

    stamps = list(first.unique())

    columns = []
    result= []
    
    for i in range(len(stamps)-1):
        start_stamp = stamps[i]
        for s in first.index:
            if ((first[s] == start_stamp) and (s not in columns)):
                columns.append(s)
        
        end_stamp = stamps[i+1]
        start_pos = list(df.index).index(start_stamp)
        end_pos = list(df.index).index(end_stamp)
        selected_index = df.index[start_pos:end_pos]
        result.append(df.loc[selected_index][columns])
    return result

second = separate_dataset(df)
        
        
        
    
    
