'''
Define the function related with the Markov Chain Monter Carlo (MCMC) process.
'''

import numpy as np
import emcee
from emcee.moves import StretchMove, DEMove, DESnookerMove, KDEMove, MHMove, WalkMove, GaussianMove
import time
import os
import git
path_git = git.Repo('.', search_parent_directories=True).working_tree_dir
path_datos_global = os.path.dirname(path_git)

def proposal_function(coords, random):
	
	"""
	Generate new proposed positions for walkers.
	Proposals are drawn from a normal distribution centered at the current position.
	"""

	step_size = 0.01  # Standard deviation of proposal distribution
	noise = random.normal(scale=step_size, size=coords.shape)  # Generate Gaussian noise
	proposal = coords + noise  # Add noise to current positions
	factors = np.ones(coords.shape[0])  # Correction factor (1 for symmetric proposals)

	return proposal, factors


def MCMC_sampler(log_probability, initial_values,
                filename = "default.h5",
                witness_file = 'witness.txt',
                max_samples = 10000,
                witness_freq = 100,
                tolerance = 0.01,
                save_path = None):
	'''
	log_probability: logarithm of the posterior distribution that will be sampled.

	initial_values: object that contains the initial value of the parameters to sample

	filename: name of the h5 file that contains the chains information.

	witness_file: name of the witness file.

	max_samples: maximun number of sample, if the chains not converge.

	witness_freq: frequency use to print the state of the calculation in the witness file.

	tolerance: tolerance parameter on the convergence method.

	save_path: directory in which the outputs are stored. Change this atribute on the

	configuration file is recommended .
	'''

	nwalkers, ndim = initial_values.shape

	# Set up the backend
	os.chdir(save_path)
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nwalkers, ndim) # Don't forget to clear it in case the file already exists
	textfile_witness = open(witness_file,'w+')
	textfile_witness.close()
	#%%
	# First, choose a move:

	## Mix moves
	#moves = [(DEMove(), 0.4), (DESnookerMove(), 0.3), (KDEMove(), 0.3)] #FIrst mix move
	#moves = [(StretchMove(), 0.5), (DEMove(), 0.3), (MHMove(np.eye(ndim) * 0.1), 0.2)] #chatgpt
	
	## Solo moves
	#moves = Walkmove() #Walkmove
	#moves = MHMove(proposal_function) #Metropolis-Hasting algorithm for a general proposal_function
	#moves = GaussianMove(np.eye(ndim) * 0.1, factor=1) #Gaussian Metropolis algorithm


	# Now, initialize the sampler:
	#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend,
	#        						moves=moves) #sample with some specify move

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend) #Dafault move



	# This will be useful to testing convergence
	old_tau = np.inf
	t1 = time.time()
	# Now we'll sample for up to max_samples steps
	for sample in sampler.sample(initial_values, iterations=max_samples, progress=True):
		# Only check convergence every 'witness_freq' steps
		if sampler.iteration % witness_freq: #'witness_freq' es cada cuanto chequea convergencia
			continue
		#print(sampler.get_log_prob()) #Print para sacar el valor del posterior en cada paso
		os.chdir(save_path)
		textfile_witness = open(witness_file,'w')
		textfile_witness.write('Iteration number: {} \t'.format(sampler.iteration))

		t2 = time.time()
		textfile_witness.write('Duration: {} minutes and {} seconds'.format(int((t2-t1)/60),
			  int((t2-t1) - 60*int((t2-t1)/60))))
		textfile_witness.close()

		# Compute the autocorrelation time so far
		# Using tol=0 means that we'll always get an estimate even
		# if it isn't trustworthy
		tau = sampler.get_autocorr_time(tol=0)

		# Check convergence
		converged = np.all(tau * 100 < sampler.iteration) #100 es el threshold de convergencia
		#También pido que tau se mantenga relativamente constante:
		converged &= np.all((np.abs(old_tau - tau) / tau) < tolerance)
		if converged:
			textfile_witness = open(witness_file,'a')
			textfile_witness.write('\n Converged!')
			textfile_witness.close()
			break
		old_tau = tau
