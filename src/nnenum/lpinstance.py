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
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0) # ADDED BACK: Silence Gurobi's console output
        env.start()

        if other_lpi is None:
            self.lp = gp.Model(env=env)
            self.lp.setParam('NumericFocus', 3) # KEPT: Prioritize numerical stability
            self.names = []
        else:
            other_lpi.deserialize()
            Timers.tic('gurobi_copy_prob')
            self.lp = other_lpi.lp.copy()
            self.names = other_lpi.names.copy()
            Timers.toc('gurobi_copy_prob')
            
        self.freeze_attrs()

    def __del__(self):
        if hasattr(self, 'lp') and self.lp is not None and not isinstance(self.lp, tuple):
            self.lp.dispose()
            self.lp = None

    def serialize(self):
        'Serialize self.lp into a tuple for multiprocessing'
        self.deserialize()
        Timers.tic('serialize')
        
        variables = self.lp.getVars()
        num_cols = len(variables)
        
        if self.lp.getObjective() is not None:
              c = self.lp.getAttr('Obj', variables)
        else:
              c = []
              
        constrs = self.lp.getConstrs()
        A_list = []
        b_list = []
        
        for constr in constrs:
            row_coeffs = np.zeros(num_cols)
            row_expr = self.lp.getRow(constr)
            for i in range(row_expr.size()):
                var = row_expr.getVar(i)
                coeff = row_expr.getCoeff(i)
                row_coeffs[var.index] = coeff
            A_list.append(row_coeffs)
            b_list.append(constr.RHS)
            
        A = np.array(A_list) if A_list else np.array([])
        b = np.array(b_list) if b_list else np.array([])
        bounds = [(v.LB, v.UB) for v in variables]

        self.lp.dispose()
        self.lp = (c, A, b, bounds, self.names)
        Timers.toc('serialize')

    def deserialize(self):
        'Deserialize self.lp from a tuple into a Gurobi model'
        if not isinstance(self.lp, tuple):
            return
        
        Timers.tic('deserialize')
        c, A, b, bounds, names = self.lp

        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0) # ADDED BACK: Silence Gurobi's console output
        env.start()
        
        self.lp = gp.Model(env=env)
        self.lp.setParam('NumericFocus', 3) # KEPT: Prioritize numerical stability
        self.names = names

        num_vars = len(bounds)
        
        if num_vars == 0:
            Timers.toc('deserialize')
            return

        var_names = names if len(names) == num_vars else [f"C{i}" for i in range(num_vars)]
        x = self.lp.addMVar(shape=num_vars, lb=[b[0] for b in bounds], ub=[b[1] for b in bounds], name=var_names)
        self.lp.update()
        
        if A.size > 0:
            self.lp.addConstr(A @ x <= b)
        
        if len(c) == num_vars:
            new_vars = self.lp.getVars()
            self.lp.setObjective(gp.LinExpr(c, new_vars), GRB.MINIMIZE)
            
        self.lp.update()
        Timers.toc('deserialize')

    def get_num_rows(self):
        self.deserialize()
        return self.lp.numConstrs

    def get_num_cols(self):
        self.deserialize()
        return self.lp.numVars
    
    def add_cols_from_names(self, names, lb, ub):
        self.deserialize()
        if not names:
            return []
        new_vars = self.lp.addVars(names, lb=lb, ub=ub)
        self.lp.update()
        self.names.extend(names)
        return new_vars.values()
        
    def add_positive_cols(self, names):
        return self.add_cols_from_names(names, lb=0.0, ub=GRB.INFINITY)
        
    def add_cols(self, names):
        return self.add_cols_from_names(names, lb=-GRB.INFINITY, ub=GRB.INFINITY)

    def add_double_bounded_cols(self, names, lb, ub):
        return self.add_cols_from_names(names, lb=float(lb), ub=float(ub))
        
    def add_dense_row(self, vec, rhs, normalize=True):
        self.deserialize()
        Timers.tic('add_dense_row')
        
        num_cols = self.get_num_cols()
        assert len(vec) == num_cols, f"Vector length {len(vec)} does not match number of columns {num_cols}"
        
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
                rhs = rhs / norm
                
        self.lp.addConstr(gp.LinExpr(vec, self.lp.getVars()) <= rhs)
        Timers.toc('add_dense_row')

    def set_minimize_direction(self, direction_vec):
        self.deserialize()
        num_cols = self.get_num_cols()
        assert len(direction_vec) == num_cols
        
        self.lp.setObjective(gp.LinExpr(direction_vec, self.lp.getVars()), GRB.MINIMIZE)

    def minimize(self, direction_vec, fail_on_unsat=True):
        self.deserialize()
        Timers.tic('gurobi_minimize')

        if self.lp.numVars > 0 and self.lp.getObjective() is None and direction_vec is None:
            self.lp.setObjective(0, GRB.MINIMIZE)
        
        if direction_vec is not None:
            self.set_minimize_direction(direction_vec)
        
        self.lp.optimize()
        
        status = self.lp.status
        rv = None

        if status == GRB.OPTIMAL:
            rv = np.array(self.lp.getAttr('X', self.lp.getVars()))
        elif status in [GRB.INFEASIBLE, GRB.UNBOUNDED]:
            rv = None 
        else:
            print(f"Gurobi finished with unhandled status: {status}")
            rv = None

        if rv is None and fail_on_unsat:
            raise UnsatError("minimize returned UNSAT and fail_on_unsat was True")
            
        Timers.toc('gurobi_minimize')
        return rv

class UnsatError(RuntimeError):
    'raised if an LP is infeasible'
