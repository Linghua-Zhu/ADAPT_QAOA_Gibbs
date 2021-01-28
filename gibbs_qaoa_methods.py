import scipy
import openfermion
import networkx as nx
import os
import numpy as np
import copy
import random
import sys
import pickle
import math
from scipy.linalg import norm

import operator_pools
from tVQE import *

from openfermion import *

def run(n,
         n_a,
         Temp,
         f,
         adapt_thresh=1e-4,
         theta_thresh=1e-7,
         layer = 1,
         pool=operator_pools.GibbsQubitPool(),
         init_para = 0.01,
         structure = 'qaoa',
         selection = 'grad',
         opt_method = 'NM'
         ):
    # {{{

    pool.init(n, n_a)
    pool.generate_SparseMatrix()

    #HA with whole dimension 2^(n + n_a)
    ham_op = pool.AB_ops[0]
    hamiltonian = pool.AB_mat[0] * 1j

 
    #HA with only half-dim, 2^n
    Ha_op = pool.sys_ops[0]
    #Hamiltonian_A = pool.sys_mat[0] * 1j
    Hamiltonian_A = openfermion.transforms.get_sparse_operator(Ha_op).toarray() 

    print('hamiltonian:',hamiltonian)


    n_ab = n + n_a

    # Start from |+> states: -->
    reference_ket = scipy.sparse.csc_matrix(
       np.full((2**n_ab, 1), 1/np.sqrt(2**n_ab))
    )

    #Start from random states: -->
    # ket_0 = np.array([[ 1.+0.j], [ 0.+0.j]])
    # ket_1 = np.array([[ 0.+0.j], [ 1.+0.j]])
    # theta_0 = np.pi*np.random.random(n_ab)
    # phi_0 = 2*np.pi*np.random.random(n_ab)
    # state_r = np.zeros((n_ab,2,1),dtype=np.complex_)
    # for i in range(0,n_ab):
    #     state_r[i] = np.cos(0.5*theta_0[i])*ket_0 + np.exp(1.0j*phi_0[i])*np.sin(0.5*theta_0[i])*ket_1
    # temp = {} # Dynamic array
    # for i in range(0, n_ab):
    #     temp[i] = np.zeros((2**(i+1), 1))
    # temp[0] = state_r[0]
    # #temp[1] = np.kron(state[0], state[1])
    # for i in range(1, n_ab):
    #     temp[i] = np.kron(temp[i-1], state_r[i])
    # ini_r = temp[n_ab-1]    

    # reference_ket = scipy.sparse.csc_matrix(ini_r)
    print('Initial states:',reference_ket)
    reference_bra = reference_ket.transpose().conj()


    # Parameters
    parameters = []

    print("Initial parameter: ", init_para)
    print(" structure :", structure)
    print(" optimizer:", opt_method)
    print(" selection :", selection)

    # Set the initial state:
    curr_state = 1.0 * reference_ket

    ansatz_ops = []
    ansatz_mat = []

    min_options = {'gtol': theta_thresh, 'disp':False}

    for p in range(0, layer):
        print(" --------------------------------------------------------------------------")
        print("                                  layer: ", p+1)
        print(" --------------------------------------------------------------------------")
        
        if structure == 'qaoa':
            ansatz_ops.insert(0, pool.AB_ops[0])
            ansatz_mat.insert(0, pool.AB_mat[0])
            parameters.insert(0, init_para)

        if selection == 'grad':
            trial_model = tUCCSD(hamiltonian, Hamiltonian_A, ansatz_mat, reference_ket, parameters, n, n_a, Temp)

            curr_state = trial_model.prepare_state(parameters)

            sig = hamiltonian.dot(curr_state)

            next_deriv = 0
            
            for op_trial in range(pool.n_ops):    

                opA = pool.spmat_ops[op_trial]
                com = 2 * (curr_state.transpose().conj().dot(opA.dot(sig))).real
                assert (com.shape == (1, 1))
                com = com[0, 0]
                assert (np.isclose(com.imag, 0))
                com = com.real

                print(" %4i %40s %12.8f" % (op_trial, pool.pool_ops[op_trial], com))

                if abs(com) > abs(next_deriv) + 1e-9:
                    next_deriv = com
                    next_index = op_trial

            new_op = pool.pool_ops[next_index]
            new_mat = pool.spmat_ops[next_index]

            print(" Add operator %4i" % next_index)

            parameters.insert(0, 0)
            ansatz_ops.insert(0, new_op)
            ansatz_mat.insert(0, new_mat)

        trial_model = tUCCSD(hamiltonian, Hamiltonian_A, ansatz_mat, reference_ket, parameters, n, n_a, Temp)

        #The current state:
        curr_state = trial_model.prepare_state(parameters)

        Free_Energy = trial_model.free_energy(parameters)

        current_energy = trial_model.energy(parameters)

        print('The current energy:', current_energy)
        print('Free energy:', Free_Energy)

        if opt_method == 'NM':    
            opt_result = scipy.optimize.minimize(trial_model.Overlap_error, parameters,
                                               method='Nelder-Mead')

        parameters = list(opt_result['x'])
        gibbs_fidelity = trial_model.Fidelity(parameters)
        error_overlap = trial_model.Overlap_error(parameters)

        print('parameters after optimization:', parameters)
        print('the overlap error:', error_overlap)
        print('The Fidelity after optimization:', gibbs_fidelity)    

     

