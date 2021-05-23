'''
Created on May 27, 2020

@author: janis
'''

#, _compute_wasserstein_distance, laplacian_kernel

from sergio.kernels import Kernel, PreprocessMixin

import numpy as np
import enum
from colito.resolvers import make_enum_resolver


class WWLLabelSource(enum.Enum):
    DATA = enum.auto()
    NONE = enum.auto()
    CORENESS = enum.auto()
    DATA_ELSE_CORENESS = enum.auto()
WWL_LABEL_SOURCE = make_enum_resolver(WWLLabelSource)
class WassersteinWeissfeilerLehmanKernel(PreprocessMixin, Kernel):
    '''
    >>> from sergio.computation import Computation
    >>> from sergio import FileManager
    >>> c = Computation(tag='test',file_manager=FileManager(paths={'data_morris':'datasets/morris'}))
    >>> c.load_dataset('morris:MUTAG').load_kernel('wwl')
    <Computation[test] D:<EntityAttributesWithGraphs[MUTAG](188x1/1)>, L:None, SG:-, M:None, O:None>
    >>> c.dataset = c.dataset.slice[5:20]
    >>> c.compute_gramian()
    <GramianFromDataset 15x15 K: <WassersteinWeissfeilerLehmanKernel()> E: <EntityAttributesWithGraphs[MUTAG](15x1/2)> P: {}>>
    '''
    __collection_title__ = 'Wasserstein Weissfeiler-Lehman Kernel'
    __collection_tag__ = 'wwl'
    
    __summary_conversions__ = {'gamma':float, 'num_iterations':int}
    __kernel_params__ = ('gamma','num_iterations','use_labels')
    
    def __init__(self, gamma:float=1., num_iterations:int=3, use_sinkhorn:bool=False, use_labels:bool=True, label_source:WWLLabelSource=WWLLabelSource.DATA_ELSE_CORENESS, verbose:bool=False):
        super().__init__()
        self.gamma:float = gamma
        self.num_iterations:int = num_iterations
        self.sinkhorn:bool = use_sinkhorn
        self.use_labels:bool = use_labels
        self.label_source = WWL_LABEL_SOURCE.resolve(label_source)
        self.verbose = verbose
        
    def parse_input(self, X):
        X_out = []
        for x in X:
            g = x.to_igraph()
            label = 'label'
            extra = []
            label_source = self.label_source
            if label_source != WWLLabelSource.NONE:
                extra = []
            else:
                if label_source in {WWLLabelSource.DATA_ELSE_CORENESS, WWLLabelSource.DATA}:
                    try:
                        nl = g.vs[label]
                    except KeyError:
                        if label_source == WWLLabelSource.DATA:
                            raise KeyError(f'While parsing WWL input: graph {x} has no vertex property "{label}".')
                        else:
                            label_source = WWLLabelSource.CORENESS
                if label_source == WWLLabelSource.CORENESS:
                    nl = g.shell_index()
                g.vs['label'] = nl
                extra = [nl]
            X_out.append((g,*extra))
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
        #from sergio.kernels.details.wwl import wwl
        graphs = [x[0] for x in X]
        
        res = wwl(graphs, node_features = None, num_iterations = self.num_iterations, sinkhorn = self.sinkhorn, gamma = self.gamma, verbose = self.verbose)
        return res
        
    def pairwise_operation(self, x, y):
        #from sergio.kernels.details.wwl import wwl
        if len(x)>1:
            xg,xn = x[:2]
            yg,yn = y[:2]
            node_features = [xn,yn]
        else:
            xg,yg = x[0],y[0]
            node_features = None
        X = [xg, yg]
        res = wwl(X, node_features = node_features, num_iterations = self.num_iterations, sinkhorn = self.sinkhorn, gamma = self.lamda, verbose=self.verbose)
        return res[0,1]


from sergio.kernels.details.wwl.src.wwl import WeisfeilerLehman, ContinuousWeisfeilerLehman
from sergio.kernels.details.wwl.src.wwl.wwl import _compute_wasserstein_distance, laplacian_kernel
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

def wwl(X, node_features=None, num_iterations=3, sinkhorn=False, gamma=None, verbose=False):
    """
    Pairwise computation of the Wasserstein Weisfeiler-Lehman kernel for graphs in X.
    """
    D_W =  pairwise_wasserstein_distance(X, node_features = node_features, 
                                num_iterations=num_iterations, sinkhorn=sinkhorn, verbose=verbose)
    res = laplacian_kernel(D_W, gamma=gamma)
    return res


if __name__ == '__main__':
    import doctest
    doctest.testmod()