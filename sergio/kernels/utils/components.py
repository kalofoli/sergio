'''
Created on Sep 7, 2021

@author: janis
'''

import typing

from colito.parallel import tqdm

from .operations import compute_entity_predicate_similarity

from sergio.kernels.utils import compute_mean_map_kernel, compute_kernel_alignment, jaccard_kernel, \
    compute_extended_tanimoto_kernel, normalise_kernel

class Validities:
    def __init__(self, V):
        self._V = V
        self._idl_ok = V.any(axis=0)
        self._K_jaccard = jaccard_kernel(self.V_ok)
        self._K_psim_ent = compute_entity_predicate_similarity(V)
        self._K_psim_set = compute_extended_tanimoto_kernel(self.V_ok, self._K_psim_ent)
    
    @property
    def K_jaccard(self):
        '''Jaccard kernel over non-empty sets'''
        return self._K_jaccard
    @property
    def K_psim_ent(self):
        '''Predicate similarity of entities'''
        return self._K_psim_ent
    @property
    def K_psim_set(self):
        '''Predicate similarity of sets'''
        return self._K_psim_set
    @property
    def K_psim_set_nrm(self):
        '''Normalised predicate similarity of sets'''
        return normalise_kernel(self.K_psim_set)
    @property
    def V(self):
        '''Set characteristic column vectors.'''
        return self._V
    @property
    def V_ok(self):
        '''Set characteristic column vectors for non-empty sets.'''
        return self._V[:,self._idl_ok]
    @property
    def n_entities(self): return self._V.shape[0]
    @property
    def n_sets(self): return self._V.shape[1]
    @property
    def n_sets_ok(self): return self._idl_ok.sum()

class BayesianAlignmentOptimiserBase:
    __tag__ = 'default'
    def __init__(self, validities, spaces, gp_opts={}):
        self._validities = validities
        self._gp_opts = {'n_calls':50, 'noise':1e-10,**gp_opts}
        self._spaces = spaces
        self._K_jac = None
        
    @property
    def spaces(self): return self._spaces
    @property
    def validities(self): return self._validities
    @property
    def V(self): return self.validities.V
    @property
    def V_ok(self): return self.validities.V_ok
    
    @property
    def alignment_target(self):
        '''The (constant) target kernel against which alignment is computed'''
        raise NotImplementedError()
    def alignment_candidate(self, K):
        '''The candidate kernel appropriately transformed so that alignment can be computed against the target kernel
        
        This is either the derived set kernel, or just the untransformed entity kernel, depending on the mode.
        '''
        raise NotImplementedError()
        
    def alignment_of_entity_kernel(self, K):
        '''Compute the alignment of a given kernel'''
        K_tar = self.alignment_target
        K_cnd = self.alignment_candidate(K)
        return compute_kernel_alignment(K_cnd, K_tar)
    
    def get_kernel(self, params):
        raise NotImplementedError()
        
    def invoke_kernel(self, idx, pars):
        X = self._features[:,[idx]]
        args, kwargs = [], {}
        it_par = iter(pars)
        for s in self._spaces:
            if s.name is None:
                args.append(next(it_par))
            else:
                kwargs[s.name] = next(it_par)
        return self._kern(X, *args, **kwargs)

    def __call__(self, with_info=False, progress=False):
        from skopt.optimizer import gp_minimize
        par_evals = {}
        def evaluate(x):
            x = tuple(x)
            
            if x not in par_evals:
                K_feat = self.get_kernel(x)
                algn = self.alignment_of_entity_kernel(K_feat)
                par_evals[x] = algn
            return -par_evals[x]
        do_opt = lambda fn: gp_minimize(fn, self.spaces, **self._gp_opts)
        if progress:
            with tqdm(total=self._gp_opts['n_calls']) as tq:
                def wrapper(*args):
                    res = evaluate(*args)
                    tq.update()
                    return res
                res = do_opt(wrapper)
        else:
            res = do_opt(evaluate)
        if with_info:
            return par_evals, res
        else:
            return par_evals

class KernelFromFeatureMixin:
    def __init__(self, *args, feature, fn_kernel: typing.Callable, param_names = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._fn_kernel: typing.Callable = fn_kernel
        self._feature = feature
        self._param_names = param_names
    
    def get_kernel(self, params):
        if self._param_names is not None:
            kwpars = dict(zip(self._param_names,params))
            pars = ()
        else:
            kwpars = {}
            pars = ()
        return self._fn_kernel(self._feature, *pars, **kwpars)

class SetKernelExtendedTanimoto:
    def set_kernel(self, K):
        return compute_extended_tanimoto_kernel(self.V_ok, K, normalise=False)
class SetKernelExtendedTanimotoNormalised:
    def set_kernel(self, K):
        return compute_extended_tanimoto_kernel(self.V_ok, K, normalise=True)
class SetKernelMeanMap:
    def set_kernel(self, K):
        return compute_mean_map_kernel(self.V_ok, K)
class SetKernelMeanMapNormalised:
    def set_kernel(self, K):
        K_set = compute_mean_map_kernel(self.V_ok, K)
        return normalise_kernel(K_set)

class PredicateAlignment:
    '''Optimiser mixin using alignment between kernels over predicates'''
    __tag__ = 'default'
    __set_kernel__ = None
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._K_set = getattr(self.validities, self.__set_kernel__)
    @property
    def K_set(self): return self._K_set
    alignment_target = K_set
    def alignment_candidate(self, K): return self.set_kernel(K)

class EntitiesAlignment:
    __entity_kernel__ = None
    '''Optimiser mixin using alignment between kernels over entities'''
    @property
    def K_ent(self): return self._K_ent
    alignment_target = K_ent
    def alignment_candidate(self, K): return K
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._K_ent = getattr(self.validities, self.__entity_kernel__)
    

class GaussianKernelFromFeatureMixin(KernelFromFeatureMixin):
    def __init__(self, *args, **kwargs):
        from sklearn.metrics.pairwise import rbf_kernel
        super().__init__(*args, fn_kernel=rbf_kernel, param_names=('gamma',), **kwargs)

