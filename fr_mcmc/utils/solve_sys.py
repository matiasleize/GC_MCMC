#%%
'''
Integration of the ODE for the different cosmological models.
'''

import os
import time

import git
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.constants import c as c_light  # units of m/s
c_light_km = c_light / 1000 # units of km/s

path_git = git.Repo(".", search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)
os.chdir(path_git); os.sys.path.append("./fr_mcmc/utils/")
from change_of_parameters import F_H_prime

#%%
def get_odes(z, Hubble, params_ode, lcdm=False):
    '''
    Returns the system of ODEs for the given cosmological model.


    Parameters:
    -----------
    z : float
        Redshift value.
    variables : list
        List of values for the dynamical variables.
    physical_params : list
        List of model parameters, where the first n-1 elements are the model parameters,
        while the last one specifies the cosmological model. Mathematically, this information is contained in
        the function Gamma.
    model : str, optional
        Cosmological model that is being integrated. Defaults to 'LCDM'.

    Returns:
    -----------
    list
        Set of ODEs for the dynamical variables.
    '''    
    #kappa = 8 * np.pi * G_newton / 3
    kappa = 1

    [lamb, L, b, L_bar, H_0, omega_m_0, model] = params_ode

    omega_r_0 = 2.47e-5
    #omega_r_0 = 4.18e-5 #2.47e-5
    rho_m_0 = 100**2 * omega_m_0 / kappa
    rho_r_0 = 100**2 * omega_r_0 / kappa
    
    a = 1/(1+z)

    rho_r = rho_r_0 * a**(-4)
    rho_m = rho_m_0 * a**(-3)
    rho_tot =  rho_r + rho_m 
    p_tot =  (1/3) * rho_r

    # To integrate in z
    s =  3 * kappa * (rho_tot + p_tot/c_light_km**2) / ((1+z)*F_H_prime(Hubble, [lamb, L, b, L_bar],model))     
    #print(s)
    return s


def integrator(physical_params, model, num_z_points=int(10**5),
                initial_z=0, final_z=3,
                system_equations=get_odes, verbose=False,
                method='RK45', rtol=1e-11, atol=1e-16):
 
    t1 = time.time()
    
    L_bar, b, H0, omega_m = physical_params
    zs_int = np.linspace(initial_z, final_z, num_z_points)
    ode_params = [0, 1e-27/H0, b, L_bar/H0, H0, omega_m, model]
    sol = solve_ivp(system_equations, (initial_z,final_z),
                    [H0], t_eval=zs_int, args = [ode_params],
                    rtol=rtol, atol=atol, method=method)
        

    #assert len(sol.t)==num_z_points, 'Something is wrong with the integration!'
    #assert np.all(zs_int==sol.t), 'Not all the values of z coincide with the ones that were required!'

    # Calculate the Hubble parameter
    zs_final = sol.t
    Hs_final = sol.y[0]

    t2 = time.time()

    if verbose == True:
        print('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
                int((t2-t1) - 60*int((t2-t1)/60))))

    return zs_final, Hs_final


def Hubble_th(physical_params, model, *args,
                z_min=0, z_max=10, **kwargs):

    '''
    Calculates the Hubble parameter as a function of redshift for different cosmological models,
    given physical parameters such as the matter density, curvature, and Hubble constant.

    Args:
        physical_params: A tuple of three physical parameters in the order (matter density, curvature, Hubble constant).
        model: A string that specifies the cosmological model to use. Valid options are 'LCDM' (Lambda-CDM),
            'BETA' (beta model) and 'GILA' (GILA model).
        b_crit: A critical value for the distortion parameter use in HS and ST models.
        all_analytic: A boolean flag that specifies whether to use an analytic approximation for the Hubble parameter
            or numerical integration.
        epsilon: A tune parameter that is used to calculate b_crit in the exponential model.
        n: An integer that specifies which of the two possible Taylor series approximations to use for the Hubble
            parameter in the power-law growth models (HS or ST).
        num_z_points: An integer that specifies the number of redshifts at which to compute the Hubble parameter.
        z_min: The minimum redshift value to consider.
        z_max: The maximum redshift value to consider.

    Returns:
        A tuple of two NumPy arrays containing the redshifts and the corresponding Hubble parameters.
    '''
    
    L_bar, b, H0, omega_m = physical_params

    zs, Hs = integrator([L_bar, b, H0,omega_m], model)  
    return zs, Hs   

#%%   
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    
    os.chdir(path_git); os.sys.path.append("./fr_mcmc/utils/")
    from LambdaCDM import H_LCDM
    
    def plot_hubble_diagram(physical_params,hubble_th=True):
        """
        Plots the Hubble diagram for a given cosmological model and physical parameters.

        Args:
            model_name (str): Name of the cosmological model to use.
            physical_params (tuple): A tuple of three physical parameters in the order (matter density, curvature, Hubble constant).
            hubble_th (bool, optional): Whether to use the Hubble function obtained from theory or numerical integration. Default is True.

        Returns:
            None. The plot is displayed in the console.

        """

        # Integrate (or evaluate) Hubble function        
        #redshifts, hubble_values = Hubble_th(physical_params) if hubble_th else \
        #                           integrator(physical_params)
        redshifts, hubble_values = integrator(physical_params)
        # Plot Hubble function
        plt.plot(redshifts, hubble_values, '.')


    # Set physical parameters
    H_0 = 70
    L_bar = 1.4 # In units of H0
    b = 0.3
    omega_m_luisa = 0.999916
    #omega_m = 0.3

    from change_of_parameters import omega_luisa_to_CDM
    omega_m = omega_luisa_to_CDM(b, L_bar, H_0, omega_m_luisa, model) #L_bar in units of H0 (inside the function L_bar is divided by H0)

    #physical_params = [L_bar, b, H_0, omega_m_luisa] #L_bar in units of H0

    #Inside the function integrator L_bar is divided by H0)

    # Plot Hubble diagrams for different models
    plt.figure()
    
    physical_params = [L_bar, b, H_0, omega_m_luisa] #L_bar in units of H0

    plot_hubble_diagram(physical_params,hubble_th=False)
    
    #Plot LCDM Hubble parameter
    redshift_LCDM = np.linspace(0,3,int(10**5))
    plt.plot(redshift_LCDM, H_LCDM(redshift_LCDM,omega_m,H_0),'k--',label=r'$\rm \Lambda CDM$') 
    
    # Format plot
    plt.title('Hubble parameter for $f(R)$ models')
    plt.xlabel('z')
    plt.ylabel(r'H(z) $\rm [(km/seg)/Mpc]$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
# %%
