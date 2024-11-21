'''
Here it is required to specify the config yml file that will be used.

TODO: see section "Ensure portability by using environment variables".
'''

from box import Box
import yaml
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
os.chdir(path_git + '/configs/')

#Here you have to specify the the name of your .yml file

##yml_file = 'config_LCDM_3p_PPS_CC.yml'

#yml_file = 'config_BETA_3p_PPS_CC_0.89.yml'
#yml_file = 'config_BETA_3p_PPS_CC_0.90.yml'
#yml_file = 'config_BETA_3p_PPS_CC_0.91.yml'

yml_file = 'config_GILA_3p_PPS_CC_0.89.yml'
#yml_file = 'config_GILA_3p_PPS_CC_0.90.yml'
#yml_file = 'config_GILA_3p_PPS_CC_0.91.yml'

#################################################
##yml_file = 'config_LCDM_3p_PPS_CC.yml'
#yml_file = 'config_LCDM_4p_PPS_CC_DESI.yml'
#yml_file = 'config_LCDM_4p_PPS_CC_BAO.yml'

##yml_file = 'config_BETA_3p_PPS_CC_0.85.yml'
#yml_file = 'config_BETA_3p_PPS_CC_0.90.yml'
##yml_file = 'config_BETA_3p_PPS_CC_0.95.yml'
#yml_file = 'config_BETA_4p_PPS_CC_DESI_0.85.yml'
#yml_file = 'config_BETA_4p_PPS_CC_DESI_0.90.yml'
#yml_file = 'config_BETA_4p_PPS_CC_DESI_0.95.yml'
#yml_file = 'config_BETA_4p_PPS_CC_BAO_0.90.yml'

##yml_file = 'config_GILA_3p_PPS_CC_0.85.yml'
#yml_file = 'config_GILA_3p_PPS_CC_0.90.yml'
##yml_file = 'config_GILA_3p_PPS_CC_0.95.yml'
#yml_file = 'config_GILA_4p_PPS_CC_DESI_0.85.yml'
#yml_file = 'config_GILA_4p_PPS_CC_DESI_0.90.yml'
#yml_file = 'config_GILA_4p_PPS_CC_DESI_0.95.yml'
#yml_file = 'config_GILA_4p_PPS_CC_BAO_0.90.yml'

#Using BAO bestfit (not useful anymore)
#yml_file = 'config_LCDM_3p_PPS_CC_DESI.yml'
#yml_file = 'config_BETA_3p_PPS_CC_DESI_0.90.yml'
#yml_file = 'config_GILA_3p_PPS_CC_DESI_0.90.yml'

#DESI data separately
#yml_file = 'config_LCDM_3p_DESI.yml'
#yml_file = 'config_BETA_3p_DESI_0.90.yml'
#yml_file = 'config_GILA_3p_DESI_0.90.yml'

#PPS data separately
#yml_file = 'config_LCDM_3p_PPS.yml'
#yml_file = 'config_BETA_3p_PPS_0.90.yml'
#yml_file = 'config_GILA_3p_PPS_0.90.yml'



with open(yml_file, "r") as ymlfile:
    full_cfg = yaml.safe_load(ymlfile)
    
cfg = Box({**full_cfg}, default_box=True, default_box_attr=None)
