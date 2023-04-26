'''
Integration of the ODE for the different cosmological models. For the Hu-Sawicki model
we use De la Cruz et al. ODE. Besides, for the Exponential model we use the Odintsov ODE.
Note that the initial conditions are different for the two models. 

TODO: Check the times of integrations of HS using De la Cruz ODE in comparison with the one of Odintsov
and evaluate the difference.

TODO: Implement Starobinsky model integration.
'''
import os
import time

import git
import numpy as np
from scipy.constants import c as c_light  # units of m/s
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from change_of_parameters import F_H, omega_luisa_to_CDM

c_light_km = c_light / 1000 # units of km/s
path_git = git.Repo(".", search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)
os.chdir(path_git)
os.sys.path.append("./fr_mcmc/utils/")
from LambdaCDM import H_LCDM
from taylor import Taylor_HS
#%%

def F_H_prime(H, params):
    lamb, L, beta, L_bar = params
    #FH_prime = 2 * H #Caso LCDM
    aux = 2 * lamb * L**6 * (lamb * (L*H)**4 + 2) * np.exp(lamb*(L*H)**4) + beta * L_bar**6 * (beta*(L_bar*H)**2 - 4) * np.exp(- beta*(L_bar*H)**2)
    FH_prime = 2 * H * (1 + H**6 * aux) 
    return FH_prime


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
        Cosmological model that is being integrated. Defaults to 'HS'.

    Returns:
    -----------
    list
        Set of ODEs for the dynamical variables.
    '''    
    #kappa = 8 * np.pi * G_newton / 3
    kappa = 1

    omega_m_0 = 0.999916
    omega_r_0 = 1 - omega_m_0

    [lamb, L, b, L_bar, H_0] = params_ode

    F_H0 = F_H(H_0, [lamb, L, b, L_bar])

    if lcdm == True:
        rho_crit_0 = H_0**2 / kappa        
    else:
        rho_crit_0 = F_H0 / kappa
    
    a = 1/(1+z)

    rho_r = rho_crit_0 * omega_r_0 * a**(-4)
    rho_m = rho_crit_0 * omega_m_0 * a**(-3)
    rho_tot =  rho_r + rho_m 
    p_tot =  (1/3) * rho_r

    # To integrate in z
    s =  3 * kappa * (rho_tot + p_tot/c_light_km**2) / ((1+z)*F_H_prime(Hubble, [lamb, L, b, L_bar]))     
    #print(s)
    return s


def integrator(physical_params, num_z_points=int(10**5),
                initial_z=0, final_z=3,
                system_equations=get_odes, verbose=False,
                method='RK45', rtol=1e-11, atol=1e-16):
 
    t1 = time.time()
    
    L_bar, b, H0 = physical_params
    zs_int = np.linspace(initial_z, final_z, num_z_points)
    ode_params = [0, 1e-27/H0, b, L_bar,H0]
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


def Hubble_th(physical_params, *args,
                z_min=0, z_max=10, **kwargs):

    '''
    Calculates the Hubble parameter as a function of redshift for different cosmological models,
    given physical parameters such as the matter density, curvature, and Hubble constant.

    Args:
        physical_params: A tuple of three physical parameters in the order (matter density, curvature, Hubble constant).
        model: A string that specifies the cosmological model to use. Valid options are 'LCDM' (Lambda-CDM),
            'EXP' (exponential model), 'HS' (Hu-Sawicki model), and 'ST' (Starobinsky model).
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
    
    L_bar, b, H0 = physical_params

    zs, Hs = integrator([L_bar, b, H0])  
    return zs, Hs   

#%%   
if __name__ == '__main__':
    from matplotlib import pyplot as plt

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
    H_0 = 73
    L_bar = 0.9/H_0
    b = 5
    physical_params_hs = np.array([L_bar, b, H_0])
    physical_params_exp = np.array([L_bar, 10, H_0])
    omega_m = omega_luisa_to_CDM(b, L_bar, H_0)

    physical_params = [L_bar, b, H_0]

    integrator(physical_params)
    # Plot Hubble diagrams for different models
    plt.figure()
    
    #for model_name, physical_params in [('HS', physical_params_hs), ('EXP', physical_params_exp)]:
    plot_hubble_diagram(physical_params,hubble_th=False)
    
    #Plot LCDM Hubble parameter
    redshift_LCDM = np.linspace(0,10,int(10**5))
    plt.plot(redshift_LCDM, H_LCDM(redshift_LCDM,omega_m,H_0),'k--',label=r'$\rm \Lambda CDM$') 
    
    # Format plot
    plt.title('Hubble parameter for $f(R)$ models')
    plt.xlabel('z')
    plt.ylabel(r'H(z) $\rm [(km/seg)/Mpc]$')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()