"""
Functions related to data management
"""

import numpy as np
from numpy.linalg import inv
import pandas as pd
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git)


class datasets():
    def __init__(self):
        pass
    #%% Pantheon plus + SH0ES
    def read_data_pantheon_plus_shoes(self, file_pantheon_plus, file_pantheon_plus_shoes_cov):

        '''
        Takes Pantheon+ data and extracts the data from the zhd and zhel 
        redshifts, its error dz, in addition to the data of the apparent magnitude
        with its error: mb and dm. With the errors of the apparent magnitude 
        builds the associated correlation matrix. The function returns the
        information of the redshifts, the apparent magnitude 
        and the correlation matrix inverse.
        '''

        # Read text with data
        df = pd.read_csv(file_pantheon_plus,delim_whitespace=True)
        df_filtered = df.filter(items=['zHD', 'm_b_corr', 'MU_SH0ES','IS_CALIBRATOR'])

        #Load the covariance matrix elements
        Ccov=np.loadtxt(file_pantheon_plus_shoes_cov,unpack=True)
        Ccov=Ccov[1:] #The first element is the total number of row/columns
        #We made the final covariance matrix..
        Ccov=Ccov.reshape(len(df['zHD']), len(df['zHD'])) # reshape with the len of the data
        #.. and finally we invert it
        Cinv=inv(Ccov)

        return df_filtered, Cinv
        #return zhd, zhel, mb, mu_shoes, Cinv, is_cal

    def read_data_pantheon_plus(self, file_pantheon_plus,file_pantheon_plus_cov):

        '''
        Takes Pantheon+ data and extracts the data from the zhd and zhel 
        redshifts, its error dz, in addition to the data of the apparent magnitude
        with its error: mb and dm. With the errors of the apparent magnitude 
        builds the associated correlation matrix. The function returns the
        information of the redshifts, the apparent magnitude 
        and the correlation matrix inverse.
        '''

        # Read text with data
        df = pd.read_csv(file_pantheon_plus,delim_whitespace=True)
        
        ww = (df['zHD']>0.01) | (np.array(df['IS_CALIBRATOR'],dtype=bool)) #mask

        #Filter columns and mask rows according to ww:
        df_filtered = df.filter(items=['zHD', 'zHEL', 'm_b_corr','IS_CALIBRATOR'])[ww]

        #Define cov mat
        Ccov=np.load(file_pantheon_plus_cov)['arr_0']
        Cinv=inv(Ccov)

        return df_filtered, Cinv
        #return zhd, zhel, Cinv, mb

    def read_data_pantheon(self, file_pantheon):

        '''
        Takes Pantheon data and extracts the data from the zcmb and zhel 
        redshifts, its error dz, in addition to the data of the apparent magnitude
        with its error: mb and dm. With the errors of the apparent magnitude 
        builds the associated correlation matrix. The function returns the
        information of the redshifts, the apparent magnitude 
        and the correlation matrix inverse.
        '''

        # Read text with data
        df = pd.read_csv(file_pantheon,delim_whitespace=True, names = ['#name','zcmb','zhel','dz','mb','dmb'],skiprows=1)
        #Create the diagonal matrx with m_B uncertainties (it depends on alpha and beta).
        Dstat=np.diag(np.power(df['dmb'].to_xarray(), 2.))

        # Read data and create the matrix with sistematic errors of NxN
        Csys=np.loadtxt('lcparam_full_long_sys.txt',unpack=True)
        Csys=Csys.reshape(len(df['zcmb']), len(df['zcmb']))
        #We made the final covariance matrix..
        Ccov=Csys+Dstat

        #.. and finally we invert it
        Cinv=inv(Ccov)
        return df, Cinv

    def read_data_chronometers(self, file_chronometers):
        # Read text with data

        df = pd.read_csv(file_chronometers,delim_whitespace=True,
                        names = ['z', 'h', 'dh'])
        return df

    def read_data_BAO(self, file_BAO):

        df = pd.read_csv(file_BAO,delim_whitespace=True,
                        names = ['z', 'data_values', 'errors_est', 'errors_sist', 'type'],
                        skiprows=1)
        df['total_errors_cuad'] = df['errors_est'].to_xarray()**2 + df['errors_sist'].to_xarray()**2
        return df

    def read_data_AGN(self, file_AGN):
        df = pd.read_csv(file_AGN,delim_whitespace=True, usecols=(3,4,5,6,7),
                        names = ['z', 'Fuv', 'eFuv', 'Fx', 'eFx'])
        df_sorted = df.sort_values(by=['z'])
        return df_sorted

    def read_data_BAO_odintsov(self, file_BAO_odintsov):
        # Read text with data
        df = pd.read_csv(file_BAO_odintsov,delim_whitespace=True,
                        names = ['z', 'h', 'dh','rd_fid'])
        return df

#%%
if __name__ == '__main__':
    import pandas as pd
    import os
    import git
    path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
    os.chdir(path_git)
    
    dfs = datasets()
    #%% Pantheon plus + SH0ES
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    df, Cinv = dfs.read_data_pantheon_plus_shoes('Pantheon+SH0ES.dat',
                                    'Pantheon+SH0ES_STAT+SYS.cov')
    #print(df,Cinv)


    #%% Pantheon plus
    os.chdir(path_git+'/fr_mcmc/source/Pantheon_plus_shoes')
    df, Cinv = dfs.read_data_pantheon_plus('Pantheon+SH0ES.dat',
                            'covmat_pantheon_plus_only.npz')
    #print(df,Cinv)

    #%% Pantheon
    os.chdir(path_git+'/fr_mcmc/source/Pantheon')
    df, Cinv  = dfs.read_data_pantheon('lcparam_full_long_zhel.txt')
    #print(df,Cinv)

    #%% AGN
    os.chdir(path_git+'/fr_mcmc/source/AGN')
    df = dfs.read_data_AGN('table3.dat')
    #print(df,Cinv)

    #%% Cosmic chronometers
    os.chdir(path_git+'/fr_mcmc/source/CC')
    df  = dfs.read_data_chronometers('chronometers_data.txt')
    #print(df,Cinv)

    #%% BAO
    os.chdir(path_git+'/fr_mcmc/source/BAO')
    file_BAO='BAO_data_da.txt'
    df = dfs.read_data_BAO(file_BAO)
    #print(df,Cinv)