# Author: Zhengyang Li
# Email: zhengyang.li@connect.polyu.hk
# Date: 2023-04-22
# Description: This file contains the community-based peer-to-peer resource-sharing model.

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog

"""
step 1: specify the input and output of the model.
    Input: willingness-to-share matrix, resources of each node, and the expected isolation days.
    Output: the resource sharing plan, represented by a n*n matrix, where n is the number of nodes.
"""

class ResourceSharingModel:
    """
    Class:
        ResourceSharingModel
    Description:
        Given the willingness-to-share matrix, the resources of each node, and the expected isolation days,
        calculate the resource sharing plan.
    """
    def __init__(self, wts_matrix:np.matrix, inventory:list, isolation_days:int):
        # step 1: specify the input and output of the model
        self.households = {}
        self.wts_matrix = wts_matrix # the willingness-to-share matrix
        self.isolation_days = isolation_days # the expected isolation days of each node
        self.inventory = np.reshape(inventory, (len(inventory), 1)) # the initial inventory vector of each node
        # Note:
        # extra_resource and needed_resource is the amount that each node can share and need.
        # shared_resource and recieved_resource is the actual amount that each node has shared and recieved.
        self.extra_resource = [] # the extra resource vector of each node
        self.needed_resource = [] # the needed resource vector of each node
        self.shared_resource = [] # the shared resource vector of each node
        self.recieved_resource = [] # the recieved resource vector of each node
        self.resource = [] # the final resource vector of each node
        self.share_plan = [] # the resource sharing plan, represented by a n*n matrix, where n is the number of nodes

    def get_extra_resource(self):
        """
        Method:
            Given the inventory vector and the expected isolation days, calculate the extra resource vector.
        Parameters:
            inventory: the inventory vector of each node.
            isolation_days: the expected isolation days of each node.
        """
        self.extra_resource = np.maximum(self.inventory - self.isolation_days, 0)

    def get_needed_resource(self):
        """
        Method:
            Given the inventory vector and the expected isolation days, calculate the needed resource vector.
        Parameters:
            inventory: the inventory vector of each node.
            isolation_days: the expected isolation days of each node.
        """
        self.needed_resource = np.maximum(self.isolation_days - self.inventory, 0)
    
    def get_shared_resource(self):
        """
        Method:
            Given the resource sharing plan, calculate the shared resource vector.
        Parameters:
            share_plan: the resource sharing plan, represented by a n*n matrix, where n is the number of nodes.
        """
        g = self.share_plan @ np.ones((np.shape(self.share_plan)[1], 1))
        self.shared_resource = g
    
    def get_recieved_resource(self):
        """
        Method:
            Given the resource sharing plan, calculate the recieved resource vector.
        Parameters:
            share_plan: the resource sharing plan, represented by a n*n matrix, where n is the number of nodes.
        """
        r = np.ones((1, np.shape(self.share_plan)[0])) @ self.share_plan
        self.recieved_resource = r.T
    
    def get_final_resource(self):
        """
        Method:
            Given the resource sharing plan, calculate the final resource vector.
        Parameters:
            share_plan: the resource sharing plan, represented by a n*n matrix, where n is the number of nodes.
        """
        self.resource = self.inventory + self.recieved_resource - self.shared_resource

    def resource_sharing_model(self):
        # This section utilize gurobi as the solver.
        """
        Method:
            Given the willingness-to-share matrix, the extra resource vector, and the needed resource vector, 
            calculate the resource sharing plan.
        Parameters:
            wts_matrix: the willingness-to-share matrix.
            extra_resource: the extra resource vector.
            needed_resource: the needed resource vector.
        """
        m = gp.Model()
        x = m.addMVar(shape=np.shape(self.wts_matrix), vtype=GRB.CONTINUOUS, name='x', lb=0)
        obj_func = 0.0
        for i in range(self.wts_matrix.shape[0]):
            obj_func += gp.quicksum(self.wts_matrix[i, :] * x[i, :])
        m.setObjective(obj_func, GRB.MAXIMIZE)
        m.addConstr(x.sum(axis=1) <= np.reshape(self.extra_resource, -1))
        m.addConstr(x.sum(axis=0) <= np.reshape(self.needed_resource, -1))
        # m.write('test.lp')
        m.optimize()
        self.share_plan = x.X

    # def resource_sharing_model(self):
    #     """
    #     Method:
    #         Given the willingness-to-share matrix, the extra resource vector, and the needed resource vector, 
    #         calculate the resource sharing plan.
    #     Parameters:
    #         wts_matrix: the willingness-to-share matrix.
    #         extra_resource: the extra resource vector.
    #         needed_resource: the needed resource vector.
    #     """
    #     n = np.shape(self.wts_matrix)[0]
    #     # step 2: define the objective function
    #     c = np.zeros(n*n)
    #     for i in range(n):
    #         for j in range(n):
    #             c[i*n+j] = self.wts_matrix[i][j]
    #     # step 3: define the constraints (sum of each row <= extra_resource, sum of each column <= needed_resource)
    #     A1 = np.zeros((n, n*n)) # sum of each row <= extra_resource
    #     for i in range(n):
    #         for j in range(n):
    #             A1[i][i*n+j] = 1
    #     A2 = np.zeros((n, n*n)) # sum of each column <= needed_resource
    #     for i in range(n):
    #         for j in range(n):
    #             A2[i][j*n+i] = 1
    #     A = np.vstack((A1, A2))
    #     b = np.vstack((self.extra_resource, self.needed_resource))
    #     # step 4: solve the linear programming problem
    #     sol = linprog(-c, A_ub=A, b_ub=b, bounds=(0, None))
    #     # step 5: post-processing
    #     plan = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(n):
    #             plan[i][j] = sol.x[i*n+j]
    #     self.share_plan = plan
    
    def solve(self):
        """
        Method:
            Solve the resource sharing problem.
        """
        # The willingness-to-share matrix should be a square matrix.
        assert np.shape(self.wts_matrix)[0] == np.shape(self.wts_matrix)[1]
        # The number of nodes should be equal to the number of rows of the willingness-to-share matrix.
        assert np.shape(self.wts_matrix)[0] == np.shape(self.inventory)[0]
        self.get_extra_resource()
        self.get_needed_resource()
        self.resource_sharing_model()
        self.get_shared_resource()
        self.get_recieved_resource()
        self.get_final_resource()