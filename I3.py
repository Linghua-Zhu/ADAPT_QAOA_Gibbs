import gibbs_qaoa_methods
import operator_pools
import numpy as np

n = 3
n_a = 3
p = 2

Temp = 0.5

filename = 'Hamiltonian' + '.txt'
f = open(filename, "a")
gibbs_qaoa_methods.run(n,
					   n_a, 
					   Temp,
	                   f,
	                   adapt_thresh=1e-4,
         			   theta_thresh=1e-7,
	                   layer=p, 
	                   pool=operator_pools.GibbsQubitPool(),
	                   init_para = 0.15,
        			   structure = 'qaoa',
                       selection = 'grad',
                       opt_method = 'NM'
	                   )
f.close()