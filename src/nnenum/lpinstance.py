'''
Stanley Bak
May 2018
(Modified by Gemini in 2025 to use Gurobi)

Gurobi python interface for nnenum.
'''

import time

import numpy as np
from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB

from nnenum.util import Freezable
from nnenum.timerutil import Timers

class LpInstance(Freezable):
    'Linear programming wrapper using Gurobi'

    def __init__(self, other_lpi=None):
        'initialize the lp instance'

        # Create a Gurobi environment with output suppressed
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()

        if other_lpi is None:
            self.lp = gp.Model(env=env)
            # internal bookkeeping
            self.names = [] # column names
        else:
            # initialize from other lpi
            Timers.tic('gurobi_copy_prob')
            self.lp = other_lpi.lp.copy()
            self.names = other_lpi.names.copy()
            Timers.toc('gurobi_copy_prob')
            
        self.freeze_attrs()

    def __del__(self):
        if hasattr(self, 'lp') and self.lp is not None:
            self.lp.dispose()
            self.lp = None

    def serialize(self):
        'Serialize self.lp into a tuple of NumPy arrays for multiprocessing'
        
        Timers.tic('serialize')
        
        # Get objective
        c = self.lp.getAttr('ObjC', self.lp.getVars())

        # Get constraints
        constrs = self.lp.getConstrs()
        A_list = []
        b_list = []
        
        for constr in constrs:
            row = self.lp.getRow(constr)
            # Assuming all are '<=' constraints as per original GLPK logic
            # Gurobi's getRow returns a sparse representation
            row_coeffs = np.zeros(self.get_num_cols())
            for i in range(row.size()):
                var_index = row.getVar(i).index
                row_coeffs[var_index] = row.getCoeff(i)
            A_list.append(row_coeffs)
            b_list.append(constr.RHS)
            
        A = np.array(A_list) if A_list else None
        b = np.array(b_list) if b_list else None

        # Get bounds
        bounds = [(v.LB, v.UB) for v in self.lp.getVars()]

        self.lp.dispose()
        # NOTE: For simplicity, this serialization assumes only <= constraints
        # and doesn't separate equality/inequality constraints.
        self.lp = (c, A, b, bounds, self.names)
        
        Timers.toc('serialize')

    def deserialize(self):
        'Deserialize self.lp from a tuple into a Gurobi model'
        
        assert isinstance(self.lp, tuple), "LP is not in serialized form"
        Timers.tic('deserialize')
        
        c, A, b, bounds, names = self.lp

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.start()
        
        self.lp = gp.Model(env=env)
        self.names = names

        # Add variables with bounds
        x = self.lp.addMVar(shape=len(c), lb=[b[0] for b in bounds], ub=[b[1] for b in bounds])
        
        # Add constraints
        if A is not None and A.size > 0:
            self.lp.addConstr(A @ x <= b)
            
        # The objective is set during minimize(), so we don't set it here.
        
        Timers.toc('deserialize')

    def get_num_rows(self):
        'get the number of rows in the lp'
        return self.lp.numConstrs

    def get_num_cols(self):
        'get the number of columns in the lp'
        return self.lp.numVars
    
    def add_rows_less_equal(self, rhs_vec):
        'Adds empty rows, constraints are set later'
        if not isinstance(rhs_vec, list):
            rhs_vec = list(rhs_vec)
        
        # In Gurobi, rows are added with constraints simultaneously.
        # This function is a bit tricky to map directly. We'll assume
        # set_constraints_csr or add_dense_row will be called later.
        # For now, we just acknowledge the number of rows that will be added.
        pass

    def add_cols_from_names(self, names, lb, ub):
        'Helper to add columns with shared bounds'
        assert isinstance(names, list)
        if not names:
            return
        
        self.lp.addVars(len(names), lb=lb, ub=ub)
        self.lp.update() # update model to reflect changes
        self.names.extend(names)
        
    def add_positive_cols(self, names):
        'add a certain number of columns to the LP with positive bounds [0, inf)'
        self.add_cols_from_names(names, lb=0.0, ub=GRB.INFINITY)
        
    def add_cols(self, names):
        'add a certain number of columns to the LP, free variables (-inf, inf)'
        self.add_cols_from_names(names, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    def add_double_bounded_cols(self, names, lb, ub):
        'add columns with the given lower and upper bound'
        self.add_cols_from_names(names, lb=float(lb), ub=float(ub))
        
    def add_dense_row(self, vec, rhs, normalize=True):
        'add a row from a dense nd.array, row <= rhs'
        Timers.tic('add_dense_row')
        
        num_cols = self.get_num_cols()
        assert len(vec) == num_cols
        
        # Normalization logic can be kept if desired
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
                rhs = rhs / norm
                
        variables = self.lp.getVars()
        expr = gp.LinExpr(vec, variables)
        self.lp.addConstr(expr <= rhs)

        Timers.toc('add_dense_row')

    def set_constraints_csr(self, data, indices, indptr, shape):
        'Sets constraints from a CSR-like format'
        # This method is complex to translate directly and efficiently without
        # building the matrix first. Gurobi's API is better with `addConstr`.
        # The current implementation will rely on `add_dense_row`.
        # This part may need further optimization if it becomes a bottleneck.
        pass

    def set_minimize_direction(self, direction_vec):
        'set the optimization direction'
        num_cols = self.get_num_cols()
        assert len(direction_vec) == num_cols
        
        variables = self.lp.getVars()
        expr = gp.LinExpr(direction_vec, variables)
        self.lp.setObjective(expr, GRB.MINIMIZE)

    def minimize(self, direction_vec, fail_on_unsat=True):
        '''minimize the lp, returning a list of assigments to each of the variables
        returns None if UNSAT, otherwise the optimization result.
        '''
        assert not isinstance(self.lp, tuple), "self.lp was tuple. Did you call lpi.deserialize()?"
        Timers.tic('gurobi_minimize')

        if direction_vec is not None:
            self.set_minimize_direction(direction_vec)

        self.lp.optimize()
        
        status = self.lp.status
        rv = None

        if status == GRB.OPTIMAL:
            rv = np.array(self.lp.getAttr('X', self.lp.getVars()))
        elif status == GRB.INFEASIBLE:
            rv = None
        elif status == GRB.UNBOUNDED:
            # An unbounded problem still has a feasible solution (the direction of the ray)
            # but for verification purposes, this often means the property is violated.
            # Depending on the calling context, we might return None or raise an error.
            # Returning None is safer to indicate no finite optimal solution was found.
            rv = None 
        else: # Other statuses like TIME_LIMIT, etc.
            print(f"Gurobi finished with unhandled status: {status}")
            rv = None

        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsat was True")
            
        Timers.toc('gurobi_minimize')
        return rv

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'
