#package loading 
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from dateutil.relativedelta import relativedelta
from scipy.stats import percentileofscore
import umap
from tqdm import tqdm
import cudf
import pickle
import numba as nb
from utility import *

#extract patient id from imported with certain medical codes, together with times
def extract_cancer_pid_time(df,\
                       extract_from_col="",\
                       code_type_col="",\
                       date_col = "",\
                       codes_icd9=np.nan,\
                       codes_icd10=np.nan,\
                       subtype=""):
    
    df_tmp = df.rename(columns={extract_from_col:"DX",code_type_col:"DX_TYPE",date_col:'DX_DATE'})
    df_tmp['DX'] = df_tmp.DX.str.replace('[^a-zA-Z0-9]', '')
    
    if (~pd.isna(codes_icd9)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['DX_TYPE'],['09','9',9])]
        search_result = df_search.DX.str.startswith(codes_icd9,na=False)
        pid_dx_9 = df_search.loc[search_result][['ID','DX','DX_TYPE','DX_DATE','ADMIT_DATE']].\
                        drop_duplicates().\
                        reset_index(drop=True)
        
    if (~pd.isna(codes_icd10)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['DX_TYPE'],['10',10])]
        search_result = df_search.DX.str.startswith(codes_icd10,na=False)
        pid_dx_10 = df_search.loc[search_result][['ID','DX','DX_TYPE','DX_DATE','ADMIT_DATE']].\
                        drop_duplicates().\
                        reset_index(drop=True)
    
    pid_dx = pd.concat([pid_dx_9,pid_dx_10])
    pid_dx['subtype'] = subtype
    return pid_dx



#extract patient id from imported with certain medical codes
def extract_cancer_pid(df,\
                       extract_from_col="",\
                       code_type_col="",\
                       codes_icd9=np.nan,\
                       codes_icd10=np.nan,\
                       subtype=""):
    
    df_tmp = df.rename(columns={extract_from_col:"DX",code_type_col:"DX_TYPE"})
    df_tmp['DX'] = df_tmp.DX.str.replace('[^a-zA-Z0-9]', '')
    
    if (~pd.isna(codes_icd9)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['DX_TYPE'],['09','9',9])]
        search_result = df_search.DX.str.startswith(codes_icd9,na=False)
        pid_dx_9 = df_search.loc[search_result][['ID','DX','DX_TYPE']].\
                        drop_duplicates().\
                        reset_index(drop=True)
        
    if (~pd.isna(codes_icd10)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['DX_TYPE'],['10',10])]
        search_result = df_search.DX.str.startswith(codes_icd10,na=False)
        pid_dx_10 = df_search.loc[search_result][['ID','DX','DX_TYPE']].\
                        drop_duplicates().\
                        reset_index(drop=True)
    
    pid_dx = pd.concat([pid_dx_9,pid_dx_10])
    pid_dx['subtype'] = subtype
    return pid_dx

#filter for records of covid19 vaccination
def extract_COVIDvax_record(df,\
                             extract_from_col="",\
                             code_type_col="",\
                             id_col="",\
                             cpt_codes=np.nan,\
                             rxnorm_codes=np.nan,\
                             ndc_codes=np.nan,\
                             cvx_codes=np.nan
                            ):
    print("Dimension before filtering: ",df.shape)
    df_tmp = df.rename(columns={extract_from_col:"VX_CODE",code_type_col:"VX_CODE_TYPE"})
    df_tmp['VX_CODE'] = df_tmp.VX_CODE.str.replace('[^a-zA-Z0-9]', '')
    
    df_result = df_tmp.loc[df_tmp['ID']=="placeholder"]
    
    if (~pd.isna(cpt_codes)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['VX_CODE_TYPE'],['CH'])]
        search_result = df_search.VX_CODE.str.startswith(cpt_codes,na=False)
        vac_cpt = df_search.loc[search_result].\
                        reset_index(drop=True)
        df_result = pd.concat([df_result,vac_cpt])
        
    if (~pd.isna(rxnorm_codes)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['VX_CODE_TYPE'],['RX'])]
        search_result = df_search.VX_CODE.str.startswith(rxnorm_codes,na=False)
        vac_rxnorm = df_search.loc[search_result].\
                        reset_index(drop=True)
        df_result = pd.concat([df_result,vac_rxnorm])
        
    if (~pd.isna(ndc_codes)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['VX_CODE_TYPE'],['ND'])]
        search_result = df_search.VX_CODE.str.startswith(ndc_codes,na=False)
        vac_ndc = df_search.loc[search_result].\
                        reset_index(drop=True)
        df_result = pd.concat([df_result,vac_ndc])
        
    if (~pd.isna(cvx_codes)):
        df_search = df_tmp.loc[is_in_set_pnb(df_tmp['VX_CODE_TYPE'],['CX'])]
        search_result = df_search.VX_CODE.str.startswith(cvx_codes,na=False)
        vac_cvx = df_search.loc[search_result].\
                        reset_index(drop=True)
        df_result = pd.concat([df_result,vac_cvx])
    print("Dimension after filtering: ",df_result.shape)
    return df_result

#EDA:aggregate by vital status and comorbidity
def cancer_subtype_summary(all_cancer_pid,death_cancer):
    all_cancer_pid = pd.merge(all_cancer_pid,death_cancer,how='left',on='ID')
    all_cancer_pid['IS_ALIVE'] = np.where(pd.isna(all_cancer_pid.DEATH_DATE),"ALIVE","DIED")
    
    subtype_summary = all_cancer_pid.groupby(['subtype','IS_ALIVE']).\
                                            agg(N_patient = ('ID','nunique')).\
                                            reset_index().\
                                            sort_values('N_patient',ascending=False).\
                                            reset_index(drop=True)
    subtype_summary = subtype_summary.pivot(index='subtype',columns='IS_ALIVE',values='N_patient')
    subtype_summary['N_pat_total'] = subtype_summary['ALIVE'] + subtype_summary['DIED']

    subtype_summary = subtype_summary.\
                                sort_values('N_pat_total',ascending=False)
   
    all_cancer_pid_pivot = pd.get_dummies(all_cancer_pid[['ID','subtype']],columns=["subtype"]).\
                        groupby('ID').\
                        max().astype(int)
    all_cancer_pid_pivot['N_subtypes'] = all_cancer_pid_pivot[list(all_cancer_pid_pivot.columns)].sum(axis=1)
    all_cancer_pid_pivot = all_cancer_pid_pivot.loc[all_cancer_pid_pivot['N_subtypes']==1].reset_index()
    all_cancer_pid_single = all_cancer_pid.loc[all_cancer_pid['ID'].\
                                               isin(all_cancer_pid_pivot['ID'])].\
                                               reset_index(drop=True)
    print("Number of patient with only one caner subtype",len(set(all_cancer_pid_single.ID)))
    cancer_subtype_single_summary = all_cancer_pid_single.groupby(['subtype']).\
                                            agg(N_patient = ('ID','nunique')).\
                                            sort_values('N_patient',ascending=False)
    cancer_subtype_single_summary = cancer_subtype_single_summary.loc[subtype_summary.index,:]
    
    subtype_summary['N_pat_MultiCancer'] = subtype_summary.N_pat_total - cancer_subtype_single_summary.N_patient
    subtype_summary['Death_rate(%)'] = np.round(subtype_summary['DIED']/subtype_summary['N_pat_total']*100,1)
    subtype_summary['MultiCancer_rate(%)'] = np.round(subtype_summary['N_pat_MultiCancer']/subtype_summary['N_pat_total']*100,1)
    return subtype_summary,all_cancer_pid


#EDA:aggregate by vital status
def cancer_subtype_cooccurrence(all_cancer_pid):
    all_cancer_pid_pivot = pd.get_dummies(all_cancer_pid[['ID','subtype']],columns=["subtype"]).\
                            groupby('ID').\
                            max().astype(int)
    all_cancer_cooccurence = all_cancer_pid_pivot.T.dot(all_cancer_pid_pivot)
    order = np.argsort(-all_cancer_cooccurence.to_numpy().diagonal())
    all_cancer_cooccurence = all_cancer_cooccurence.iloc[order, order]
    
    return all_cancer_cooccurence


#EDA:number of vaccine records
def cancer_subtype_vax(all_cancer_pid,vax_cancer):
    vax_num_agg = vax_cancer.groupby('ID').\
                    agg(N_vax_record = ('VX_ID','nunique')).\
                    reset_index().\
                    sort_values('N_vax_record',ascending=False).\
                    reset_index(drop=True)
    vax_num_agg['N_vax_record_cap'] = np.minimum(vax_num_agg['N_vax_record'],4)
    
    #ADJUST VAX COUNTS BY DOES CODES, IF AVAILABLE
    cpt_codes_4th = tuple(['0004A','0034A','0044A','0054A','0064A',
                           '0074A','0094A','0104A','0124A','0134A',
                           '0144A','0154A','0164A'])
    cpt_codes_3rd = tuple(['0003A','0013A','0053A','0073A','0083A',
                           '0093A','0113A','0173A'])
    cpt_codes_2nd = tuple(['0002A','0012A','0022A','0042A','0052A',
                           '0072A','0082A','0092A','0112A'])
    
    id_4th = set(vax_cancer.loc[is_in_set_pnb(vax_cancer['VX_CODE'],cpt_codes_4th)].ID)
    id_3rd = set(vax_cancer.loc[is_in_set_pnb(vax_cancer['VX_CODE'],cpt_codes_3rd)].ID)
    id_2nd = set(vax_cancer.loc[is_in_set_pnb(vax_cancer['VX_CODE'],cpt_codes_2nd)].ID)
    
    vax_num_agg['N_vax_record_adj']=np.where(is_in_set_pnb(vax_num_agg['ID'],id_4th),
                4,
                    np.where(is_in_set_pnb(vax_num_agg['ID'],id_3rd),
                    3,
                        np.where(is_in_set_pnb(vax_num_agg['ID'],id_2nd),
                        2,
                        vax_num_agg['N_vax_record_cap']
                        )   
                    )
                )
    vax_num_agg['N_vax'] = np.maximum(vax_num_agg['N_vax_record_cap'],vax_num_agg['N_vax_record_adj'])
    
    all_cancer_pid_vax = pd.merge(all_cancer_pid,vax_num_agg,on='ID',how='left')
    all_cancer_pid_vax['N_vax'] = all_cancer_pid_vax['N_vax'].fillna(0).astype(int)
    
    all_cancer_pid_vax_non0 = all_cancer_pid_vax.loc[all_cancer_pid_vax['N_vax']>0].reset_index()
    print("Number of COVIDwithCancer paitent vaccinated: "+str(len(set(all_cancer_pid_vax_non0['ID'])) )+" out of "+str(len(set(all_cancer_pid_vax['ID']))))
    print("Percentage of COVIDwithCancer paitent with vaccination data(%): "+ str(np.round(100*len(set(all_cancer_pid_vax_non0['ID']) )/len(set(all_cancer_pid_vax['ID'])),2)))
    
    all_cancer_pid_vax = pd.get_dummies(all_cancer_pid_vax,columns=["N_vax"])    
    
    all_cancer_pid_vax_tmp = all_cancer_pid_vax[['ID','subtype','N_vax_0','N_vax_1',\
                                             'N_vax_2','N_vax_3','N_vax_4']].drop_duplicates()
    all_cancer_pid_vax_agg = all_cancer_pid_vax_tmp.groupby('subtype').\
                                                agg(N_pat = ('ID','nunique'),
                                                    N_vax_0 = ('N_vax_0','sum'),
                                                    N_vax_1 = ('N_vax_1','sum'),
                                                    N_vax_2 = ('N_vax_2','sum'),
                                                    N_vax_3 = ('N_vax_3','sum'),
                                                    N_vax_4 = ('N_vax_4','sum')).\
                                                sort_values('N_pat',ascending=False).\
                                                rename(columns={'N_vax_4':'N_vax_noless4'})
    all_cancer_pid_vax_agg['N_vaxed'] = (all_cancer_pid_vax_agg['N_pat'] - all_cancer_pid_vax_agg['N_vax_0']).astype(int)
    
    return all_cancer_pid_vax_agg, all_cancer_pid_vax,vax_num_agg