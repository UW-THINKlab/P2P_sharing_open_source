# Author: Zhengyang Li
# Email: zhengyang.li@connect.polyu.hk
# Email: lzy95@uw.edu
# Date: 2023-06-08
# Description: This file defines the objects and methods for a community.


import numpy as np
import networkx as nx


class Community(nx.Graph):
    """
    This class inherits the nx.Graph class to define the community object. 
    The community is a undirected graph, where each node is a household.
    The edges are the social ties between households.

    To create a community object, we can use the following code:
    community = Community()

    To add nodes and edges to the community, we can use the following code:
    community.add_nodes_from([0,1,2,3,4])
    community.add_edges_from([(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (2,3), (3,4)])

    To add attributes to the nodes, we can use the following code:
    community.nodes[1]['name'] = 'household1'
    community.nodes[2]['name'] = 'household2'
    community.nodes[1]['transp'] = 7
    community.nodes[2]['transp'] = 5

    To add attributes to the edges, we can use the following code:
    community.edges[1,2]['weight'] = 3
    community.edges[1,2]['type'] = "strong_tie"
    
    After creating the nodes, to generate the resource inventory of each household, we can use the following code:
    community.generate_resource_inventory(amount=[0,1,2,3,4], pmf=[0.1, 0.2, 0.3, 0.2, 0.2])

    To generate the sharing preference of each household, we can use the following code:
    community.generate_sharing_preference(recipient=[0,1,2,3], pmf=[0.1, 0.2, 0.3, 0.4])
    """
    def __init__(self) -> None:
        super().__init__()

    def set_resource_inventory(self, resource_inventory:list, name='transp'):
        """
        This function sets the resource inventory of each household.
        Parameters:
            resource_inventory: a list of resource inventory.
            name: the name of the resource.
        """
        assert len(resource_inventory) == len(self.nodes()) # The length of amount_list should be equal to the number of households.
        for ind, household in enumerate(self.nodes()):
            self.nodes[household][name] = resource_inventory[ind]
    
    def generate_resource_inventory(self, support:list, pmf:list, name='transp', seed=0):
        """
        This function generates the resource inventory of each household.
        Parameters:
            support: possible values for resource inventory.
            pmf: probability mass w.r.t. values in the value_set.
            name: the name of the resource.
            seed: the seed of the random generator.
        """
        assert abs(1 - sum(pmf)) <= 1e-12 # The sum of the probability list should be 1.
        assert len(support) == len(pmf) # The length of amount_list should be equal to the length of p_list.
        np.random.seed(seed)
        for household in self.nodes():
            self.nodes[household][name] = np.random.choice(support, p=pmf)

    def get_resource_inventory(self, name='transp') -> list:
        """
        This function returns the resource inventory of each household.
        """
        resource_inventory = []
        for household in self.nodes():
            resource_inventory.append(self.nodes[household][name])
        return resource_inventory

    def set_sharing_preference(self, recipient:list, name="transp_share_preference"):
        """
        This function sets the sharing preference of each household.
        Parameters:
            recipient: list, the list of the recipient of sharing.
        """
        for ind, household in enumerate(self.nodes()):
            self.nodes[household][name] = recipient[ind]

    def generate_sharing_preference(self, recipient:list, pmf:list, name="transp_share_preference", seed=0):
        """
        Method:
            This function generates the sharing preference of each household.
        Parameters:
            recipient: list, the list of the recipient of sharing.
        """
        assert abs(sum(pmf) - 1) <= 1e-12  # The sum of the probability list should be 1.
        assert len(recipient) == len(pmf) # The length of amount_list should be equal to the length of p_list.
        np.random.seed(seed)
        for household in self.nodes():
            self.nodes[household][name] = np.random.choice(recipient, p=pmf)

    def get_sharing_preference(self, name="transp_share_preference") -> list:
        """
        This function returns the sharing preference of each household.
        """
        sharing_preference = []
        for household in self.nodes():
            sharing_preference.append(self.nodes[household][name])
        return sharing_preference

    def generate_corrlated_sharing_preference(self, recipient:list, pmf_mat:np.matrix, cov_mat:np.matrix, 
                                              name:list=["water_share_preference",
                                                         "food_share_preference",
                                                         "meds_share_preference",
                                                         "transp_share_preference",
                                                         "comm_share_preference",
                                                         "faid_share_preference",
                                                         "warm_share_preference",
                                                         "sani_share_preference",
                                                         "power_share_preference",
                                                         "shelter_share_preference"], 
                                                         seed=0):
        """
        Method:
            This function generates the sharing preference of each household.
        Parameters:
            recipient: list, the list of the recipient of sharing.
            pmf_mat: the probability mass function matrix. Each row is the pmf of a resource.
            cov_mat: the covariance matrix of the resources.
            name: the name of the sharing preference.
            seed: the seed of the random generator.     
        """
        
        sum_pmf = pmf_mat.sum(axis=1)
        assert np.abs(sum_pmf - np.ones_like(sum_pmf)) <= 1e-12  # The sum of the probability list should be 1.
        assert len(recipient) == len(pmf_mat.shape[1]) # The length of amount_list should be equal to the number of columns in pmf_mat.
        assert cov_mat.shape[0] == cov_mat.shape[1] # asset cov_mat is a square matrix
        assert np.all(np.linalg.eigvals(cov_mat) >= 0) # asset cov_mat is a positive semi-definite matrix
        assert len(name) == cov_mat.shape[0] # the length of name should be equal to the number of resources

        uncorr_sharing_preferences = np.zeros((pmf_mat.shape[0], len(self.nodes())))

        np.random.seed(0)
        for i in range(uncorr_sharing_preferences.shape[0]):
            for j in range(uncorr_sharing_preferences.shape[1]):
                uncorr_sharing_preferences[i, j] = np.random.choice(recipient, p=pmf_mat[i, :])
        
        corr_sharing_preferences = np.linalg.cholesky(cov_mat) @ uncorr_sharing_preferences
        # To be updated. The outcome is not integer.
        return 0

    def generate_social_ties(self, degrees:list, distance_matrix:np.matrix, distance_decay_alpha:float=-1.35, seed:int=0):
        """
        This function generates the social ties of the community.
        """
        assert len(degrees) == len(self.nodes())
        assert distance_matrix.shape[0] == len(self.nodes())
        assert distance_matrix.shape[1] == len(self.nodes())
        np.random.seed(seed)

        # make sure the sum of the degrees is even
        degrees = np.array(degrees)
        if degrees.sum() % 2 != 0:
            degrees[0] += 1

        # set the diagonal of the distance matrix to be infinity
        # so that the distance between a household and itself is infinity
        np.fill_diagonal(distance_matrix, np.inf)

        # calculate the probability of having a tie
        prob_matrix = self._distance_decay_function(distance_matrix, alpha=distance_decay_alpha)

        # generate the social network
        social_tie_matrix = np.zeros(prob_matrix.shape, dtype=int)
        accumulated_degree_list = np.zeros(len(degrees), dtype=int)

        for i in range(np.shape(prob_matrix)[0]):
            remain_degree = degrees - accumulated_degree_list # the number of remaining degree for each household
            if remain_degree[i] <= 0:
                continue
            reachable_households = np.where(remain_degree > 0)[0] # the households that are reachable from household i
            select_probability = prob_matrix[i, reachable_households]/np.sum(prob_matrix[i, reachable_households])
            selected_households = np.random.choice(reachable_households, 
                                            size=remain_degree[i], 
                                            p=select_probability,
                                            replace=True)
            # To be strict, replace should be False. However, in that case, the degree 
            # requirement is very likely not satisfied. This problem is a standalone 
            # network configuration problem, which is not the focus of this paper.
            social_tie_matrix[i, selected_households] = 1
            social_tie_matrix[selected_households, i] = 1
            accumulated_degree_list = np.sum(social_tie_matrix, axis=1)

        self.add_edges_from(np.argwhere(social_tie_matrix==1))
        
    def _distance_decay_function(self, distance, alpha=-1.35):
        """
        Method:
            The distance decay function is used to calculate the probability of two households having a tie.
        Parameters:
            distance: the distance between two households
            alpha: the decay rate
        return:
            the probability of two households having a tie
        Reference:
            alpha values:
            (alpha = -1.2) Liben-Nowell, D., Novak, J., Kumar, R., Raghavan, P., Tomkins, A., 2005. Geographic routing in social networks. 
                Proceedings of the National Academy of Sciences 102, 11623–11628.
            (alpha = -0.74) Xu, Y., Santi, P., & Ratti, C. (2022). Beyond distance decay: Discover homophily in spatially embedded social networks. 
                Annals of the American Association of Geographers, 112(2), 505-521.
        """
        eps = 1e-12 # to avoid division by zero
        return (distance + eps) ** alpha
    
    def split_social_ties(self, types:list, pmf:list, seed=0):
        """
        This function splits the social ties into different types.
        """
        assert sum(pmf) == 1
        assert len(pmf) == len(types)
        np.random.seed(seed)

        for tie in self.edges():
            self.edges[tie]['type'] = np.random.choice(types, p=pmf)

    def get_social_tie_matrix(self, type:str=None) -> np.array: 
        """
        This function returns the social tie matrix.
        """
        if type is None:
            return nx.adjacency_matrix(self).todense()
        else:
            # return the adjacency matrix of edges with the given type
            matrix = np.zeros((len(self.nodes()), len(self.nodes())), dtype=int)
            for tie in self.edges():
                if self.edges[tie]['type'] == type:
                    matrix[tie[0], tie[1]] = 1
                    matrix[tie[1], tie[0]] = 1
            return matrix

    def get_share_priority_matrix(self, sharing_preference:str, priority=[3, 2, 1]):
        """
        Method:
            Generate the willingness to share matrix.
        Parameters:
            sharing_preference: the name of sharing preference
            priority: the priority of sharing (for link weight)
        Return:
            priority_matrix: the sharing priority matrix
        """
        strong_tie_matrix = self.get_social_tie_matrix(type='strong')
        weak_tie_matrix = self.get_social_tie_matrix(type='weak')
        
        n = strong_tie_matrix.shape[0]
        stranger_matrix = np.ones(shape=(n, n)) - self.get_social_tie_matrix()

        # create willingness to share matrix
        priority_matrix = np.zeros((n, n))

        for n in self.nodes():
            share_preferece_val = self.nodes[n][sharing_preference]
            if share_preferece_val == 0: # no sharing
                priority_matrix[n, :] = (0 * strong_tie_matrix[n, :])
            if share_preferece_val == 1: # share with strong ties
                priority_matrix[n, :] = (priority[0] * strong_tie_matrix[n, :])
            if share_preferece_val == 2: # share with strong ties and weak ties
                priority_matrix[n, :] = (priority[0] * strong_tie_matrix[n, :]
                                        + priority[1] * weak_tie_matrix[n, :])
            if share_preferece_val == 3: # share with strong ties, weak ties and strangers
                priority_matrix[n, :] = (priority[0] * strong_tie_matrix[n, :]
                                        + priority[1] * weak_tie_matrix[n, :]
                                        + priority[2] * stranger_matrix[n, :])
        return priority_matrix
    
    def get_share_network(self, sharing_preference, priority=[3, 2, 1]):
        """
        Method:
            Construct the share network based on the sharing preference and social ties.
        Parameters:
            sharing_preference: the name of sharing preference
            priority: the priority of sharing (for link weight)
        Return:
            G: the share network
        """
        priority_matrix = self.get_share_priority_matrix(sharing_preference, priority)
        G = nx.from_numpy_array(priority_matrix, create_using=nx.DiGraph)
        # set the node attributes
        for n in G.nodes():
            G.nodes[n].update(self.nodes[n])
        return G
    