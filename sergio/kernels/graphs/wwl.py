'''
Created on May 27, 2020

@author: janis
'''

from wwl import ContinuousWeisfeilerLehman, WeisfeilerLehman
from wwl.wwl import _compute_wasserstein_distance, laplacian_kernel

from . import Kernel

import numpy as np

def pairwise_wasserstein_distance(X, node_features = None, num_iterations=3, sinkhorn=False, enforce_continuous=False, verbose=False):
    """
    Pairwise computation of the Wasserstein distance between embeddings of the 
    graphs in X.
    args:
        X (List[ig.graphs]): List of graphs
        node_features (array): Array containing the node features for continuously attributed graphs
        num_iterations (int): Number of iterations for the propagation scheme
        sinkhorn (bool): Indicates whether sinkhorn approximation should be used
    """
    # First check if the graphs are continuous vs categorical
    categorical = True
    if enforce_continuous:
        if verbose:
            print('Enforce continous flag is on, using CONTINUOUS propagation scheme.')
        categorical = False
    elif node_features is not None:
        if verbose:
            print('Continuous node features provided, using CONTINUOUS propagation scheme.')
        categorical = False
    else:
        for g in X:
            if not 'label' in g.vs.attribute_names():
                if verbose:
                    print('No label attributed to graphs, use degree instead and use CONTINUOUS propagation scheme.')
                categorical = False
                break
        if categorical and verbose:
            print('Categorically-labelled graphs, using CATEGORICAL propagation scheme.')
    
    # Embed the nodes
    if categorical:
        es = WeisfeilerLehman()
        node_representations = es.fit_transform(X, num_iterations=num_iterations)
    else:
        es = ContinuousWeisfeilerLehman()
        node_representations = es.fit_transform(X, node_features=node_features, num_iterations=num_iterations)

    # Compute the Wasserstein distance
    pairwise_distances = _compute_wasserstein_distance(node_representations, sinkhorn=sinkhorn, 
                                    categorical=categorical, sinkhorn_lambda=1e-2)
    return pairwise_distances

def wwl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None):
    """
    Pairwise computation of the Wasserstein Weisfeiler-Lehman kernel for graphs in X.
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations, sinkhorn=sinkhorn)
    res = laplacian_kernel(D_W, gamma=gamma)
    return res

class WassersteinWeissfeilerLehmanKernel(Kernel):
    name = 'Wasserstein Weissfeiler-Lehman Kernel'
    tag = 'wasserstein-weissfeiler-lehman'
    
    _summary_convert = {'lamda':float, 'num_iterations':int}
    def __init__(self, lamda:float=1., num_iterations:int=3, sinkhorn:bool=False, n_jobs:int=None, normalize:bool=False, verbose:bool=False, use_labels:bool=True, label_with_coreness:bool=False):
        super().__init__(n_jobs=n_jobs, normalize=normalize,verbose=verbose)
        self.lamda:float = lamda
        self.num_iterations:int = num_iterations
        self.sinkhorn:bool = sinkhorn
        self.use_labels:bool = use_labels
        self.label_with_coreness:bool = label_with_coreness
        
    def parse_input(self, X):
        from grakel.graph import Graph
        import igraph
        X_out = []
        for x in X:
            g = Graph(*x)
            E = g.get_edges()
            x_ig = igraph.Graph(E)
            extra = []
            if self.use_labels:
                if self.label_with_coreness:
                    nl = x_ig.shell_index()
                else:
                    nl_dct = g.get_labels('vertex')
                    nl = np.array([nl_dct[i] for i in range(len(nl_dct))])
                x_ig.vs['label'] = nl
                extra.append(nl)
            X_out.append((x_ig,*extra))
        return X_out
        
    def fit_transform(self, X):
        self.fit(X)
        X = self.X
        K = self.wwl(X)
        return K
    
    def transform(self, Y):
        Y_parsed = self.parse_input(Y)
        n = len(self.X)
        XY = [*self.X,*Y_parsed]
        Kxy = self.wwl(XY)
        K = Kxy[n:,:n]
        return K

    def wwl(self, X):
        graphs = [x[0] for x in X]
        
        res = wwl(graphs, node_features = None, num_iterations = self.num_iterations, sinkhorn = self.sinkhorn, gamma = self.lamda)
        return res
        
    
    def pairwise_operation(self, x, y):
        if len(x)>1:
            xg,xn = x[:2]
            yg,yn = y[:2]
        else:
            xg,yg = x[0],y[0]
        X = [xg, yg]
        node_features = [xn,yn]
        res = wwl(X, node_features = node_features, num_iterations = self.num_iterations, sinkhorn = self.sinkhorn, gamma = self.lamda)
        return res[0,1]

