#package loading 
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from dateutil.relativedelta import relativedelta
from scipy.stats import percentileofscore
#import umap
from tqdm import tqdm
import cudf
import pickle
import numba as nb


#helper functions
def min_max_date(df,vocab_date):
    df_tmp = df
    df_tmp[vocab_date] = pd.to_datetime(df_tmp[vocab_date])
    col_name_min = 'min_'+str(vocab_date)
    col_name_max = 'max_'+str(vocab_date)
    col_name_na = 'na_prop_'+str(vocab_date)
    df_tmp = df.groupby('ID').agg(Min = (vocab_date,'min'),
                                  Max = (vocab_date,'max'),
                                  na_prop = (vocab_date,lambda x: x.isnull().mean())).reset_index()
    df_tmp.columns = ['ID',col_name_min,col_name_max,col_name_na]
    return df_tmp

#define a faster version of np.isin(.)
@nb.jit(parallel=True)
def is_in_set_pnb(a, b):
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)