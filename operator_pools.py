import openfermion
import numpy as np
import copy as cp

from openfermion import *


class OperatorPool:
    def __init__(self):
        self.n = 0
        self.n_a = 0
        self.n_AB = 0

    def init(self, n, n_a):

        self.n = n
        self.n_a = n_a
        self.n_AB = n + n_a

        #ham = openfermion.ops.QubitOperator()
        #for i in range(self.n_system)
        #    ham -= openfermion.ops.QubitOperator("Z%d Z%d" % (i, (i + 1) % n_chain))

        self.generate_SQ_Operators()

    def generate_SparseMatrix(self):
        self.sys_mat = []
        self.ani_mat = []
        self.AB_mat = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.sys_ops:
            self.sys_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n))
        for op in self.ani_ops:
            self.ani_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n_a))
        for op in self.AB_ops:
            self.AB_mat.append(transforms.get_sparse_operator(op, n_qubits=self.n_AB))
        # print(self.sys_mat[0]) 

        self.spmat_ops = []
        for op in self.pool_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits=self.n_AB))
        return

class GibbsQubitPool(OperatorPool):
    # def init(self, n, n_a):
    #     #self.register_len = register_len
    #     #self.n_registers = n_registers
    #     #self.n_spin_orb = n_registers * (register_len)
    #     self.n_system = n 
    #     self.n_ancilla = n_a
    #     self.n_AB = n + n_a 

    #     self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        """
        Creates a set of operators for doing Gibbs state preparation with
        ADAPT-QAOA. For free energy estimation, we need a number of copies of the
        same underlying system, so our pool needs to consist only of operators
        which affect all copies the same way, and which only allow a given copy
        to interact with itself. We also assume time reversal symmetry, so we
        only need sums of Pauli strings with odd numbers of Y operators. We
        include all such single- and two-qubit Pauli strings.
        """
        self.sys_ops = []
        self.ani_ops = []
        self.AB_ops = []
        self.pool_ops = []



        # Now we set the problem Hamiltonian A, which is like cost Hamiltonian 
        AB = QubitOperator('Z0 Z1', 0)
        for i in range(self.n_a, self.n_AB):
            AB -= QubitOperator("Z%d Z%d" % (i, (i + 1) % self.n + self.n_a))
        self.AB_ops.append(AB)

        A = QubitOperator('Z0 Z1', 0)
        for i in range(self.n):
            A -= QubitOperator("Z%d Z%d" % (i, (i + 1) % self.n))
        self.sys_ops.append(A)
            
        # Now we set the ancilla & problem Hamiltonian H_AB, which is like mixer Hamiltonian
        # single qubit operators
        Y = QubitOperator('Y0', 0)
        X = QubitOperator('X0', 0)

        for i in range(self.n_AB):
            X += QubitOperator('X%d' % i, 1j) 
        self.AB_ops.append(X)
        self.pool_ops.append(X)

        for i in range(self.n_AB):
            Y += QubitOperator('Y%d' % i, 1j)
        self.AB_ops.append(Y)
        self.pool_ops.append(Y)


        # 2-qubit operators
        YZ = QubitOperator('Y0 Z1', 0)
        ZY = QubitOperator('Y0 Z1', 0)
        XY = QubitOperator('X0 Y1', 0)
        YX = QubitOperator('Y0 X1', 0)
        XX = QubitOperator('X0 X1', 0)
        YY = QubitOperator('Y0 Y1', 0)



        for i in range(0, self.n):
            for j in range(i + self.n, self.n_AB):
                YZ = QubitOperator('Y%d Z%d' % (i, j), 1j)
                self.AB_ops.append(YZ)
                self.pool_ops.append(YZ)
                ZY = QubitOperator('Z%d Y%d' % (i, j), 1j)
                self.AB_ops.append(ZY)
                self.pool_ops.append(ZY)
                XY = QubitOperator('X%d Y%d' % (i, j), 1j)
                self.AB_ops.append(XY)
                self.pool_ops.append(XY)
                YX = QubitOperator('Y%d X%d' % (i, j), 1j)
                self.AB_ops.append(YX)
                self.pool_ops.append(YX)
                XX = QubitOperator('X%d X%d' % (i, j), 1j)
                self.AB_ops.append(XX)
                self.pool_ops.append(XX)
                YY = QubitOperator('Y%d Y%d' % (i, j), 1j)
                self.AB_ops.append(YY)
                self.pool_ops.append(YY)

        self.n_ops = len(self.pool_ops)

    def get_string_for_term(self, op):
        """
        Gets a string representation of any given operator
        @param op The operator to convert
        @returns A string describing op
        """
        return " + ".join([" ".join(["%s%d" % (ti[1], ti[0]) for ti in t]) for t in op.terms])
        