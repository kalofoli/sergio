'''
Created on Oct 7, 2019

@author: janis
'''

import numpy as np
import grakel

from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg import cg
from scipy.linalg import expm
from numpy.linalg import multi_dot
from numpy.linalg import inv
import scipy.sparse.linalg as sprla 

from cofi.kernel_detail import Kernel, ShiftInvariantKernel
from typing import Iterable
from sklearn.utils import Bunch
from grakel.graph import Graph as gkGraph
from cofi.logging import getModuleLogger

import graph_tool as gt
from abc import ABCMeta, abstractmethod
from cofi.factory import NoConversionError, parse_constructor_string
from cofi.kernel_detail.euclidean import IndicatorSIKernel
from cofi.utils.statistics import StatisticsBase, StatisticsUpdater
from cofi.utils import StatsBunch
from collections import namedtuple

log = getModuleLogger(__name__)

class RandomWalkStatistics(StatisticsBase):
    increase_mv_computations = StatisticsUpdater('mv_computations', 1, doc='Matrix Vector Computations')
    increase_pair_computations = StatisticsUpdater('pair_computations', 1, doc='Graph-pair Kernel Computations')
    increase_solver_invocations = StatisticsUpdater('solver_invocations', 1, doc='Graph-pair Kernel Computations')
    
class RandomWalkKernel(grakel.RandomWalk, Kernel):
    name = 'Random Walk'
    tag = 'random-walk'

    _summary_compact_fields = ['kernel_type','lamda']
    
    def __init__(self, n_jobs=None,
                 normalize=False, verbose=False,
                 lamda=0.1, method_type="fast",
                 kernel_type="geometric", p=None):
        super().__init__(n_jobs=n_jobs,
                 normalize=normalize, verbose=verbose,
                 lamda=lamda, method_type=method_type,
                 kernel_type=kernel_type, p=p)
        self._stats = RandomWalkStatistics()

    def stats_block(self):
        '''this is a context. Use with "with"'''
        return self._stats.block()
    
    def pairwise_operation(self, X, Y):
        """Calculate the random walk kernel.

        Fast:
        Spectral demoposition algorithm as presented in
        :cite:`vishwanathan2006fast` p.13, s.4.4, with
        complexity of :math:`O((|E|+|V|)|E||V|^2)` for graphs witout labels.

        Baseline:
        Algorithm presented in :cite:`kashima2003marginalized`,
        :cite:`gartner2003graph` with complexity of :math:`O(|V|^6)`

        Parameters
        ----------
        X, Y : Objects
            Objects as produced from parse_input.

        Returns
        -------
        kernel : number
            The kernel value.

        """
        if self.method_type == "baseline":
            # calculate the product graph
            XY = np.kron(X, Y)

            # algorithm presented in
            # [Kashima et al., 2003; Gartner et al., 2003]
            # complexity of O(|V|^6)

            # XY is a square matrix
            s = XY.shape[0]

            if self.p is not None:
                P = np.eye(XY.shape[0])
                S = self.mu_[0] * P
                for k in self.mu_[1:]:
                    P = np.matmul(P, XY)
                    S += k*P
            else:
                if self.kernel_type == "geometric":
                    S = inv(np.identity(s) - self.lamda*XY).T
                elif self.kernel_type == "exponential":
                    S = expm(self.lamda*XY).T

            value = np.sum(S)
        elif self.method_type == "fast" and (self.p is not None or self.kernel_type == "exponential"):
            # Spectral demoposition algorithm as presented in
            # [Vishwanathan et al., 2006] p.13, s.4.4, with
            # complexity of O((|E|+|V|)|E||V|^2) for graphs
            # witout labels

            # calculate kernel
            qi_Pi, wi = X
            qj_Pj, wj = Y

            # calculate flanking factor
            ff = np.expand_dims(np.kron(qi_Pi, qj_Pj), axis=0)

            # calculate D based on the method
            Dij = np.kron(wi, wj)
            if self.p is not None:
                D = np.ones(shape=(Dij.shape[0],))
                S = self.mu_[0] * D
                for k in self.mu_[1:]:
                    D *= Dij
                    S += k*D

                S = np.diagflat(S)
            else:
                # Exponential
                S = np.diagflat(np.exp(self.lamda*Dij))
            value = ff.dot(S).dot(ff.T)
        else:
            # Random Walk
            # Conjugate Gradient Method as presented in
            # [Vishwanathan et al., 2006] p.12, s.4.2
            Ax, Ay = X, Y
            xs, ys = Ax.shape[0], Ay.shape[0]
            mn = xs*ys

            def lsf(x, lamda):
                xm = x.reshape((xs, ys), order='F')
                y = np.reshape(multi_dot((Ax, xm, Ay)), (mn,), order='F')
                self._stats.increase_mv_computations()
                return x - self.lamda * y

            # A*x=b
            A = LinearOperator((mn, mn), matvec=lambda x: lsf(x, self.lamda))
            b = np.ones(mn)
            x_sol, _ = cg(A, b, tol=1.0e-6, maxiter=20, atol='legacy')
            value = np.sum(x_sol)
        self._stats.increase_pair_computations()
        return value

import scipy.sparse as spr

def tkron_times_vector(A,B,x,w_krn):
    na,nb = A.shape[0], B.shape[0]

    vec = lambda X:X.flatten(order='F')
    mat = lambda x:x.reshape(nb,na,order='F')
    
    T = mat(w_krn)
    
    y = T*(B@(T*mat(x))@A.T)
    return vec(y)

def get_kron_size(ga, gb, d):
    ka = len(ga)
    kb = len(gb)
    diff = np.repeat(ga, kb) - np.tile(gb, ka)
    kron_size = np.sum(np.abs(diff)<=d)
    return kron_size

class SparseGraph(Bunch):
    pass

class TruncatedRandomWalkKernel(Kernel, metaclass=ABCMeta):
    '''Abstract class: Truncated random walk'''
    _summary_compact_fields = ['kernel_type','lamda','vertex_kernel','bandwidth']
    _SolverResult = namedtuple('_SolverResult',('value','n_solver'))
    
    def __init__(self, n_jobs:int=None, normalize:bool=False, verbose:bool=False,
                 lamda:float=0.1, method_type:str="fast",kernel_type:str="geometric", p:int=None,
                 vertex_kernel:ShiftInvariantKernel = IndicatorSIKernel(),
                 tolerance:float=1e-6, cg_max_iters:int=20, truncate:bool=False):
        self.kernel_type = kernel_type
        self.method_type = method_type
        self.lamda = lamda
        self.vertex_kernel:ShiftInvariantKernel = vertex_kernel
        self.p = p
        self.tolerance:float = tolerance 
        self.cg_max_iters:int = cg_max_iters
        self._stats = RandomWalkStatistics()
        self.truncate:bool=truncate
        super().__init__(verbose=verbose, n_jobs=n_jobs, normalize=normalize)

    @property
    def stats(self): return StatsBunch(**self._stats._asdict())

    @classmethod
    def parse_string_argument(cls, name, value, parameter):
        if name == 'vertex_kernel':
            cls_name, args, kwargs = parse_constructor_string(value)
            vk = ShiftInvariantKernel.make_from_strings(cls_name, *args, **kwargs)
            return vk
        raise NoConversionError()

    def stats_block(self):
        '''this is a context. Use with "with"'''
        return self._stats.block()
    
    @abstractmethod
    def vertex_groups(self, sg:SparseGraph) -> np.ndarray:
        '''Return an array of vertex indices from the processed graph'''
        pass

    def process_entry(self, g:gkGraph):
        # import graph_tool as gt
        from scipy.sparse import csc_matrix

        sg = SparseGraph()

        # A = sg.A = g.get_adjacency_matrix()
        E = g.get_edges('adjacency')
        IJ = tuple(zip(*E))
        n = sg.n = g.nv()
        m = sg.m = len(IJ[0])
        sg.S = csc_matrix((np.ones(m),IJ),shape=(n,n))
        try:
            nl = g.get_labels('vertex')
            nl = np.array([nl[i] for i in range(g.nv())])
        except ValueError:
            nl = None
        sg.node_labels = nl
        self.process_sparse_entry(g, sg)
        return sg
        
    
    @abstractmethod
    def process_sparse_entry(self, g:gkGraph, sg:SparseGraph):
        '''Extend SparseGraph with data necessary to later retrieve indices.'''
        pass
    
    def compute_rw_baseline(self, sx, sy, w):
        # AA is a square matrix
        AA_full = spr.kron(sx.S,sy.S,'csr')
        AA = AA_full.multiply(w[None,:]).multiply(w[:,None])
        n = AA.shape[0]

        if self.p is not None:
            k = np.arange(1,self.p+1)
            l = self.lamda
            if self.kernel_type == 'exponential':
                mu1 = l**k/np.cumprod(k)
            else:
                mu1 = l**k
            mu0 = 1
            P = spr.eye(n)
            S = mu0*P # Technically this is mu0, but mu0 is always 1
            for k in range(1,self.p+1):
                P *= AA
                S += mu1[k-1]*P
        else:
            if self.kernel_type == "geometric":
                S = inv(np.eye(n) - self.lamda*AA).T
            elif self.kernel_type == "exponential":
                S = expm(self.lamda*AA).T

        return self._SolverResult(value=np.sum(S), n_solver=n)

    def get_matrix_vector_op(self, X, Y):
        Ax, Ay = X.S, Y.S
        w = self.get_weights_kron(X, Y)
        def mv(x):
            self._stats.increase_mv_computations()
            y = tkron_times_vector(Ax,Ay,x,w)
            return y
        n = X.n*Y.n
        return mv, n

    def get_weights_kron(self, X, Y):
        grp_x = self.vertex_groups(X)
        grp_y = self.vertex_groups(Y)
        
        nx = len(grp_x)
        ny = len(grp_y)
        
        grp_y_krn = np.kron(np.ones(nx),grp_y)
        grp_x_krn = np.kron(grp_x,np.ones(ny))
        grp_diff = grp_y_krn-grp_x_krn
        w_krn = self.vertex_kernel(grp_diff)
        return w_krn
        

    def compute_rw_fast_geometric(self, sx, sy):
        ''' mvn is the matrix-vector operator and its size n'''
        # Random Walk
        # Conjugate Gradient Method as presented in
        # [Vishwanathan et al., 2006] p.12, s.4.2

        mv,n = self.get_matrix_vector_op(sx, sy)
        
        if n>0:
            def lsf(x):
                y = mv(x)
                return x - self.lamda * y
            
            # A*x=b
            A = LinearOperator((n, n), matvec=lsf)
            b = np.ones(n)
            x_sol, _ = cg(A, b, tol=self.tolerance, maxiter=self.cg_max_iters)
            res = np.sum(x_sol)
            self._stats.increase_solver_invocations()
        else:
            res = 0
        return self._SolverResult(value=res, n_solver=n)

    def compute_rw_fast_exponential(self, sx, sy):
        from .expm import expm_multiply, ConstantOperations
        # Higham's algorithm as presented in
        # ??
        # complexity of O((|E|+|V|)|E||V|^2) for graphs
        # without labels
    
        mv,n = self.get_matrix_vector_op(sx, sy)
        if n>0:
            # TODO: use tighter norm bounds exploiting T?
            A = LinearOperator((n, n), matvec=mv, rmatvec=mv)
            b = np.ones(n)
            
            lo_norm1 = lambda : sprla.norm(sx.S,1)*sprla.norm(sx.S,1)
            lo_norminf = lambda : sprla.norm(sx.S,np.inf)*sprla.norm(sx.S,np.inf)
            sptrace = lambda S: S.diagonal().sum()
            lo_trace = lambda : sptrace(sx.S)*sptrace(sy.S)*self.vertex_kernel(0)
            x_sol = expm_multiply(A, b, t = self.lamda, tol=self.tolerance,co = ConstantOperations(A, trace=lo_trace, norm1=lo_norm1, norminf=lo_norminf))
            res = np.sum(x_sol)
            debug = False
            if debug:
                # TODO: removeme
                y=b.copy()
                z=b
                f=1
                for i in range(1,20):
                    z=self.lamda*mv(z)
                    f *=i
                    y+=z/f
                nrm = np.linalg.norm(x_sol-y)
                if nrm > 1e-3:
                    print(f'Check this: norm deviation: {nrm}')
            self._stats.increase_solver_invocations()
        else:
            res = 0
        return self._SolverResult(value=res, n_solver=n)
        

    def pair_bandwidth(self, sx, sy):
        bw_max = max(sx.n, sy.n)
        vk = self.vertex_kernel
        if self.bandwidth is None:
            if vk.bandwidth is None:
                d = bw_max
            else:
                d = vk.bandwidth
        else:
            d = self.bandwidth
        d = min(d, bw_max)
        return d

    def pair_kron_size(self, sx, sy):
        gx = self.vertex_groups(sx)
        gy = self.vertex_groups(sy)
        d = self.pair_bandwidth(sx, sy)
        kron_size = get_kron_size(gx, gy, d)
        return kron_size

    def pair_truncation_offset(self, sx, sy, solver_result:_SolverResult):
        '''This function computed the result offset arising from dimensional differences between solver and kernel.
        
        Some solvers use only the truncated part, while some the full dimensions. In both kernel types (exp/geom)
        if the dimensions differ a constant offset is observed:
        For the geometric, the entries of inv(I-lAx) which correspond to the zero A elements do not appear in the sum, but do appear in the I matrix.
        Simailarly for the exponential one.
        
        We must therefore here add/remove their effect.'''
        n_full = sx.n*sy.n
        n_solver = solver_result.n_solver
        is_truncated = n_full != n_solver
        if self.truncate ^ is_truncated:
            # correction needed
            if is_truncated:
                # add the missing entries
                n_kernel = sx.n*sy.n
            else:
                n_kernel = self.pair_kron_size(sx, sy)
            offset = n_kernel - n_solver
        else:
            offset = 0
        return offset

    def pairwise_operation(self, sx, sy):
        """Calculate a pairwise kernel between two elements.

        Parameters
        ----------
        x, y : Object
            Objects as occur from parse_input.

        Returns
        -------
        kernel : number
            The kernel value.

        """
        
        if self.method_type == "baseline":
            w_krn = self.get_weights_kron(sx, sy)
            solver_result = self.compute_rw_baseline(sx, sy, w_krn)
        elif self.method_type == "fast":
            if self.kernel_type == "exponential":
                solver_result = self.compute_rw_fast_exponential(sx, sy)
            elif self.kernel_type == 'geometric':
                solver_result = self.compute_rw_fast_geometric(sx, sy)
            else:
                raise NotImplementedError(f'Kernel type {self.kernel_type} is not implemented. Available are: [exponential, geometric]')
        else:
            raise NotImplementedError(f'Method type {self.method_type} is not implemented. Available are: [fast, baseline]')
        
        offset = self.pair_truncation_offset(sx, sy, solver_result)
        result = solver_result.value + offset
        self._stats.increase_pair_computations()
        return result

    def parse_input(self, X):
        """Parse and create features for random_walk kernel.

        Parameters
        ----------
        X : iterable
            For the input to pass the test, we must have:
            Each element must be an iterable with at most three features and at
            least one. The first that is obligatory is a valid graph structure
            (adjacency matrix or edge_dictionary) while the second is
            node_labels and the third edge_labels (that correspond to the given
            graph format). E valid input also consists of graph type objects.

        Returns
        -------
        out : list
            The extracted adjacency matrices for any given input.

        """
        if not isinstance(X, Iterable):
            raise TypeError('input must be an iterable\n')
        else:
            i = 0
            out = list()
            for (idx, x) in enumerate(X):
                is_iter = isinstance(x, Iterable)
                if is_iter:
                    x = list(x)
                if is_iter and len(x) in [0, 1, 2, 3]:
                    if len(x) == 0:
                        log.warn(f'Found empty graph on index: {idx}.')
                    node_labels = x[1] if len(x)>1 else {}
                    edge_labels = x[2] if len(x)>2 else {}
                    g = gkGraph(x[0], node_labels, edge_labels, 'dictionary')
                elif type(x) is gkGraph:
                    g = x
                else:
                    raise TypeError('each element of X must be either a ' +
                                    'graph or an iterable with at least 1 ' +
                                    'and at most 3 elements\n')
                i += 1
                out.append(self.process_entry(g))

            if i == 0:
                raise ValueError('parsed input is empty')

            return out

class IntegerRandomWalkKernel(TruncatedRandomWalkKernel):

    _summary_compact_fields = ['kernel_type','lamda']

    def __init__(self, n_jobs:int=None, normalize:bool=False, verbose:bool=False,
                 lamda:float=0.1, method_type:str="fast",kernel_type:str="geometric", p:int=None,
                 mv_method:str= 'numpy', bandwidth:int = None,
                 vertex_kernel:ShiftInvariantKernel = IndicatorSIKernel(),
                 tolerance:float=1e-6, cg_max_iters:int=20, truncate:bool=False):
        
        self.mv_method:str = mv_method
        self.bandwidth:int = bandwidth
        self.truncate:bool = truncate
        super().__init__(n_jobs=n_jobs, normalize=normalize, verbose=verbose,
                 lamda=lamda, method_type=method_type,kernel_type=kernel_type, p=p,
                 vertex_kernel=vertex_kernel,
                 tolerance=tolerance, cg_max_iters=cg_max_iters, 
                 truncate=truncate)

    def vertex_groups(self, g)->np.ndarray:
        raise NotImplementedError()
    
    def get_matrix_vector_op(self, X, Y):
        mv_method = self.mv_method
        if mv_method == 'numpy':
            mv, n = super().get_matrix_vector_op(X, Y)
        elif mv_method == 'cpp' or mv_method == 'cppsparse':
            import pycofi_impl as pi
            vk = self.vertex_kernel
            d = self.pair_bandwidth(X, Y)
            grp_a = self.vertex_groups(X)
            grp_b = self.vertex_groups(Y)

            w_grp = vk(np.arange(-d, d+1))
            if mv_method == 'cpp':
                gmX = pi.SortedGrouppedMatrix(X.S.todense(), grp_a)
                gmY = pi.SortedGrouppedMatrix(Y.S.todense(), grp_b)
                
                tkm = pi.TruncatedKronMatrix(gmX, gmY, w_grp, d)
            else:
                gmX = pi.SparseSortedGrouppedMatrix(X.S.todense(), grp_a)
                gmY = pi.SparseSortedGrouppedMatrix(Y.S.todense(), grp_b)
                
                tkm = pi.SparseTruncatedKronMatrix(gmX, gmY, w_grp, d)
            n = tkm.kron_size
            y = np.empty(n)
            def mv(x):
                self._stats.increase_mv_computations()
                if x.ndim > 1:
                    x = x.flatten()
                tkm.multiply_trunc_vector(x, y)
                return y
        else:
            raise ValueError(f'Only methods "numpy", "cpp" and "ccpsparse" are available for matrix-vector computations.')
        return mv, n

class NLRandomWalkKernel(IntegerRandomWalkKernel):
    name = 'Node-Labelled Random Walk'
    tag = 'nl-random-walk'

    def vertex_groups(self, g)->np.ndarray:
        return g.node_labels
    
    def process_sparse_entry(self, g:gkGraph, sg:SparseGraph):
        pass

class CoreRandomWalkKernel(IntegerRandomWalkKernel):
    name = 'Core Random Walk'
    tag = 'core-random-walk'

    def vertex_groups(self, g)->np.ndarray:
        return g.kcd
    
    def process_sparse_entry(self, g:gkGraph, sg:SparseGraph):
        from graph_tool import Graph as gtGraph
        from graph_tool.topology import kcore_decomposition
        
        gtg = gtGraph(directed=False)
        gtg.add_vertex(g.nv())
        gtg.add_edge_list(g.get_edges(purpose='adjacency'))
        sg.kcd = np.array(kcore_decomposition(gtg).a)

class DegreeRandomWalkKernel(IntegerRandomWalkKernel):
    name = 'Degree Random Walk'
    tag = 'degree-random-walk'

    def vertex_groups(self, g)->np.ndarray:
        return g.degrees
    
    def process_sparse_entry(self, g:gkGraph, sg:SparseGraph):
        from graph_tool import Graph as gtGraph
        
        gtg = gtGraph(directed=False)
        gtg.add_vertex(g.nv())
        gtg.add_edge_list(g.get_edges(purpose='adjacency'))
        sg.degrees = np.array(gtg.degree_property_map('total').a)

class WLRandomWalkKernel(TruncatedRandomWalkKernel):
    name = 'Weisfeiler Lehman Random Walk'
    tag = 'wl-random-walk'

    _summary_compact_fields = ['kernel_type', 'lamda', 'wl_iterations']


    def __init__(self, n_jobs:int=None, normalize:bool=False, verbose:bool=False,
                 lamda:float=0.1, method_type:str="fast",kernel_type:str="geometric", p:int=None,
                 tolerance:float=1e-6, cg_max_iters:int=20, wl_iterations:int = 1, truncate:bool=False):
        
        self.bandwidth: int = 0
        self.wl_iterations = wl_iterations
        super().__init__(n_jobs=n_jobs, normalize=normalize, verbose=verbose,
                 lamda=lamda, method_type=method_type,kernel_type=kernel_type, p=p,
                 tolerance=tolerance, cg_max_iters=cg_max_iters,truncate=truncate)

    def vertex_groups(self, sg)->np.ndarray:
        return sg.wl_labels
    
    def process_sparse_entry(self, g, sg):

        from graph_tool import Graph as gtGraph
        
        gtg = gtGraph(directed=False)
        gtg.add_vertex(g.nv())
        gtg.add_edge_list(g.get_edges(purpose='adjacency'))
        sg.wl_labels = WLLabels(gtg, self.wl_iterations)

def WLLabels(g: gt.Graph, n_iter: int = 1, labels: np.ndarray = None):
    '''
    Compute Weisfeiler Lehman labels for the vertices of an unlabeled graph g.

    returns the weisfeiler lehman vertex labels of iteration n_iter in a numpy array.
    '''
    # convenience: call this without label parameter and it still works
    if labels is None:
        labels = np.zeros(g.num_vertices(), np.int64)
    if n_iter == 0:
        return labels

    # we need an immutable copy of the old labels
    old_labels = labels
    labels = np.zeros(old_labels.shape, np.int64);

    for i in range(g.num_vertices()):
        # collect and sort neighbor labels from prev iteration
        nl = old_labels[g.get_all_neighbors(i)]
        nl.sort()
        # create new label string (in a numpy array for speed)
        wl = np.hstack([old_labels[i], nl])
        # hash label array to int value. fingers crossed that there is no collision!
        labels[i] = hash(wl.data.tobytes())

    # recurse to implement higher iterations
    return WLLabels(g, n_iter-1, labels)


class MIRandomWalkKernel(IntegerRandomWalkKernel):
    name = 'Morgan Index Random Walk'
    tag = 'mi-random-walk'

    _summary_compact_fields = IntegerRandomWalkKernel._summary_compact_fields + ['mi_iterations']


    def __init__(self, n_jobs:int=None, normalize:bool=False, verbose:bool=False,
             lamda:float=0.1, method_type:str="fast",kernel_type:str="geometric", p:int=None,
             mv_method:str= 'numpy', bandwidth:int = None, mi_iterations:int=5,
             vertex_kernel:ShiftInvariantKernel = IndicatorSIKernel(),
             tolerance:float=1e-6, cg_max_iters:int=20, truncate:bool=False):
        
        self.mi_iterations = mi_iterations
        super().__init__(n_jobs=n_jobs, normalize=normalize, verbose=verbose,
             lamda=lamda, method_type=method_type,kernel_type=kernel_type, p=p,
             mv_method=mv_method, bandwidth=bandwidth,
             vertex_kernel=vertex_kernel, tolerance=tolerance,
             cg_max_iters=cg_max_iters, truncate=truncate)

    def vertex_groups(self, sg)->np.ndarray:
        return sg.mi_labels
    
    def process_sparse_entry(self, g, sg):

        from graph_tool import Graph as gtGraph
        
        gtg = gtGraph(directed=False)
        gtg.add_vertex(g.nv())
        gtg.add_edge_list(g.get_edges(purpose='adjacency'))
        sg.mi_labels = MILabels(gtg, self.mi_iterations)

def MILabels(g: gt.Graph, n_iter: int = 1, labels: np.ndarray = None):
    '''
    Compute Morgan Index labels for the vertices of an unlabeled graph g.

    returns the Morgan index vertex labels of iteration n_iter in a numpy array.
    '''
    # convenience: call this without label parameter and it still works
    if labels is None:
        labels = np.zeros(g.num_vertices(), np.int64)
    if n_iter == 0:
        return labels

    # we need an immutable copy of the old labels
    old_labels = labels
    labels = np.zeros(old_labels.shape, np.int64);

    for i in range(g.num_vertices()):
        # collect and sort neighbor labels from prev iteration
        nl = old_labels[g.get_all_neighbors(i)]
        nl.sort()
        # create new label string (in a numpy array for speed)
        wl = np.hstack([old_labels[i], nl])
        # hash label array to int value. fingers crossed that there is no collision!
        labels[i] = hash(wl.data.tobytes())

    # recurse to implement higher iterations
    return WLLabels(g, n_iter-1, labels)


if __name__ == '__main__':
    from cofi.experiment import Experiment
    e = Experiment()
    e.load_dataset('graph:MUTAG')
    
    k = CoreRandomWalkKernel()
    X = e.dataset.data[:10]
    k.fit(X)
    K = k.transform(e.dataset.data);
    print(K)
    
    
    
    
    