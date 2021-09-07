'''
Created on Sep 7, 2021

@author: janis
'''


from sergio.kernels.utils.components import \
    GaussianKernelFromFeatureMixin, EntitiesAlignment, PredicateAlignment, \
    SetKernelExtendedTanimoto, BayesianAlignmentOptimiserBase,\
    SetKernelExtendedTanimotoNormalised, SetKernelMeanMap


__all__ = [
    'ParametricKernelOptimiserEntities',
    'ParametricKernelOptimiserSetsExtendedTanimotoPredicateSimilarity',
    'ParametricKernelOptimiserSetsExtendedTanimotoNormalisedPredicateSimilarity',
    'ParametricKernelOptimiserSetsMeanMapJaccard',
    'ParametricKernelOptimiserSetsMeanMapPredicateSimilarity',
    'ParametricKernelOptimiserSetsMeanMapNormalisedPredicateSimilarity'
]


class ParametricKernelOptimiserEntities(GaussianKernelFromFeatureMixin, EntitiesAlignment, BayesianAlignmentOptimiserBase):
    __tag__ = 'ent_psim'
    __entity_kernel__ = 'K_psim_ent'
class ParametricKernelOptimiserSetsExtendedTanimotoPredicateSimilarity(
    GaussianKernelFromFeatureMixin, PredicateAlignment,
    SetKernelExtendedTanimoto, BayesianAlignmentOptimiserBase):
    __tag__ = 'set_et_psim'
    __set_kernel__ = 'K_psim_set'
class ParametricKernelOptimiserSetsExtendedTanimotoNormalisedPredicateSimilarity(
    GaussianKernelFromFeatureMixin, PredicateAlignment,
    SetKernelExtendedTanimotoNormalised, BayesianAlignmentOptimiserBase):
    __tag__ = 'set_etn_psim'
    __set_kernel__ = 'K_psim_set_nrm'
class ParametricKernelOptimiserSetsMeanMapJaccard(
    GaussianKernelFromFeatureMixin, PredicateAlignment,
    SetKernelMeanMap, BayesianAlignmentOptimiserBase):
    __tag__ = 'set_mm_jac'
    __set_kernel__ = 'K_jaccard'
class ParametricKernelOptimiserSetsMeanMapPredicateSimilarity(
    GaussianKernelFromFeatureMixin, PredicateAlignment,
    SetKernelMeanMap, BayesianAlignmentOptimiserBase):
    __tag__ = 'set_mm_psim'
    __set_kernel__ = 'K_psim_set'
class ParametricKernelOptimiserSetsMeanMapNormalisedPredicateSimilarity(
    GaussianKernelFromFeatureMixin, PredicateAlignment,
    SetKernelMeanMap, BayesianAlignmentOptimiserBase):
    __tag__ = 'set_mmn_psim'
    __set_kernel__ = 'K_psim_set_nrm'



