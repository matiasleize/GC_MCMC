# GILA model statistical analysis
Author: Matías Leizerovich. Faculty of Exact and Natural Sciences, Buenos Aires University.

For download, see https://github.com/matiasleize/GILA_MCMC

You can use GILA_MCMC freely, provided that in your publications you cite the paper mentioned.

## Overview
This project is designed to read and process various types of astronomical data, including chronometers, BAO, and DESI data. It also includes functionality to plot the data for visualization.


## Create a virtual environment
In order to create a virtual environment with the libraries that are needed to run this module, follow the next steps:
* Clone the repository: ``` git clone``` 
* Enter the directory: ```cd GILA_MCMC```
* Create the virtual environment: ```conda env create``` 
* Activate the virtual environment: ```source activate fR-MCMC```

## Create an output directory:
Output files can be particarly heavy stuff. For instance, the markov chains are saved in h5 format of several MegaBites. To avoid the unnecessary use of memory on the main repository, output files are stored on an independent directory on the computer's user. For default, this file must be created on the same directory that the Git's repository was cloned:

```
root_directory/                Root directory
├── GILA_MCMC/                 Root project directory
├── GILA-output/               Output directory
```

Having said that, the user can change the location of the ouput directory on the configuration file.

## Configuration file:
The files (.yml) located on the directory fR-MCMC/configs shows all the configuration parameters. 

## Run the code:
To run the code for a particular configuration file, edit config.py (which is located on the directory fR-MCMC/fr_mcmc) and then run the following command while you are on the root project directory:  

```
python3 -m fr_mcmc --task mcmc
```

If it is desired to run only the analyses part of the code over an existing Markov Chain result, run:

```
python3 -m fr_mcmc --task analysis --outputfile 'filename'
```

where 'filename' is the name of the directory where the runs are stored (as an example: 'filename' =  'sample_GILA_SN_CC_4params').

## Running Tests
To run the tests, use the following command:

```
python -m unittest discover -s tests
```

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details.