'''
Created on Jun 2, 2019

@author: janis
'''

import numpy as np
import graph_tool as gt
from numpy import ndarray
from typing import NamedTuple, Dict, Any

from sdcore.logging import getLogger, FinerLogger
from sdcore.utils import RemainingIndexMapper
from sdcore.summarisable import Summarisable, SummaryOptions
from sdcore.language import Language
from collections import OrderedDict
#from sdcore.utils import StatisticsMeta, StatisticsBase

log = getLogger(__name__.split('.')[-1])

#class Statistics(StatisticsBase):
#    popped = StatisticsMeta.Counter(0)
    
#    @StatisticsMeta.updater(queued, step=1)
#    def increase_popped(self): pass


class GalbrunCommunities(Summarisable):
    
    class _Community(NamedTuple):
        density_remain: float
        density_full: float
        features:ndarray
        vertices: ndarray
        edges: ndarray
        
    class Community(_Community, Summarisable):
        def __str__(self):
            return f'<{self.__class__.__name__} |V|={len(self.vertices)}, |E|:{len(self.edges)} |F|:{len(self.features)} density w/remain edges: {self.density_remain} w/ all edges: {self.density_full}>'
        @property
        def num_vertices(self):
            return len(self.vertices)
        
        def summary_dict(self, options:SummaryOptions) -> Dict[str, Any]:
            '''The parameters to be included in the summary as a dict'''
            summary = OrderedDict((
                ('density_remain',float(self.density_remain)), 
                ('density_full',float(self.density_full)), 
                ('features',tuple(map(int, self.features)))
                ))
            return summary
        
    class CommunityCollection(Summarisable.List):
        pass
  
    def __init__(self, data, language: Language, k:int=1) -> None:
        self._language: Language = language
        self._k = k
        self._communities: GalbrunCommunities.CommunityCollection = None
        
        self._g = data.asgraph(attributes=())
        self._validities = np.stack(tuple(p.validate() for p in self._language.predicates),1)
        
        self._gt_e_available = self._g.new_edge_property('bool')
        self._gt_validity = self._g.new_vertex_property('bool')
        
    @property
    def graph(self):
        return self._g
    
    @property
    def k(self):
        return self._k
    
    @property
    def validities(self):
        return self._validities
    
    @property
    def communities(self):
        return self._communities
    
    def subgroups(self):
        from sdcore.language import ConjunctionSelector
        def make_selector(community):
            selector = ConjunctionSelector(self._language, community.features)
            return selector
        return tuple(map(make_selector, self._communities))
        
    def _get_feature_densities(self, idx_feats, v_remain, e_remaining):
        gt_validity = self._gt_validity
        gt_e_available = self._gt_e_available
        
        def eval_density(idx_feat):
            gt_validity.a[:] = v_remain & self._validities[:, idx_feat]
            gv = gt.GraphView(self._g, vfilt=gt_validity, efilt=gt_e_available)
            return np.float64(gv.num_edges()) / gv.num_vertices()
        
        gt_e_available.a[:] = e_remaining
        return np.fromiter(map(eval_density, idx_feats), float)
    
    def get_induced_edges(self, validities):
        gt_validity = self._gt_validity
        gt_validity.a[:] = validities
        gv = gt.GraphView(self._g, vfilt=gt_validity)
        e = gv.get_edges()
        return e[:,2]
        
        
    def residual_densest_mapped_unsafe_untested(self, e_covered, e_comm_id, communities):
        class CommunityCandidate(NamedTuple):
            density: float
            feature: int
            
        num_features = self._validities.shape[1]
        num_vertices = self._g.num_vertices()
        e_remaining = ~e_covered
        
        map_feat = RemainingIndexMapper(num_features)
        p_dens = np.zeros(num_features)
        
        community_sizes = np.fromiter((c.num_vertices for c in communities), int)
        
        num_edges_remaining = e_remaining.sum()
        v_remain = np.ones(num_vertices, bool)
        comm_candidates = []
        while num_edges_remaining and map_feat:
            # Remove redundant features: Those which do not further limit the current validity
            is_feat_redund = np.where(np.all(self._validities[np.ix_(v_remain,map_feat.remain)],0))[0]
            #print(f"removng: {is_feat_redund} {map_feat[:]} {map_feat(None)}")
            removal = map_feat.remove_many(is_feat_redund)
            removal.apply_numpy(p_dens)
            #print(f"removed: {is_feat_redund} {map_feat[:]} {map_feat(None)}")
            if log.rlim.debug:
                log.debug(f'Removed fetures: {removal.idx_ful} (idx_rem:{is_feat_redund}).')
            if not map_feat:
                break
    
            p_dens_rem = self._get_feature_densities(map_feat.remain, v_remain, e_remaining)
            p_dens[:len(map_feat)] = p_dens_rem
            
            idx_feat_rem_max = np.nanargmax(p_dens_rem)
            #print(f"removng: {idx_feat_rem_max} {map_feat[:]} {map_feat(None)}")
            removal = map_feat.remove(idx_feat_rem_max)
            #print(f"removng: {idx_feat_rem_max} {map_feat[:]} {map_feat(None)}")
            cur_dens = removal.apply_sequence(p_dens)
            comm_candidates.append(CommunityCandidate(density=float(cur_dens), feature=int(removal.idx_ful)))
            
            cur_validity = self._validities[:,removal.idx_ful]
            v_remain &= cur_validity
            cur_num_vertices = v_remain.sum()
            
            num_edges_remaining_greedy = e_remaining.sum()
            if communities:
                e_idx_induced = self.get_induced_edges(v_remain)
                e_comm_size_exceeded = community_sizes[e_comm_id[e_idx_induced]] > cur_num_vertices
                e_remaining[e_idx_induced[e_comm_size_exceeded]] = True
            if log.rlim.progress:
                cnt_feats = map_feat.num_full-len(map_feat)
                log.progress((f'Added feat: {removal.idx_ful:4} dens: {cur_dens:6.4} (add/del/tot:{len(comm_candidates)}/{cnt_feats}/{map_feat.num_full}) '
                              f'cover {cur_num_vertices}/{num_vertices} ({(cur_num_vertices/num_vertices)*100:5.1f}%). '
                              f'Stolen {num_edges_remaining-num_edges_remaining_greedy} edges.'))
            num_edges_remaining = e_remaining.sum() if communities else num_edges_remaining_greedy
        
        candidate_densities = np.fromiter((c.density for c in comm_candidates),float)
        idx_best_cand = np.nanargmax(candidate_densities)
        best_density = candidate_densities[idx_best_cand]
        best_features = np.fromiter((c.feature for c in comm_candidates[:idx_best_cand+1]), int)
        v_densest_bool = np.any(validities[:,best_features], axis=1)
        e_densest = self.get_induced_edges(v_densest_bool)
        v_densest = np.where(v_densest_bool)[0]
        community = self.Community(density=best_density, vertices=v_densest, edges=e_densest, features=best_features)
        return community
    
    def residual_densest(self, e_covered, e_comm_id, communities):
        class CommunityCandidate(NamedTuple):
            density: float
            feature: int
            
        index_bool2int = lambda x: np.where(x)[0]
        
        num_features = self._validities.shape[1]
        num_vertices = self._g.num_vertices()
        
        e_remain = ~e_covered
        f_remain = np.ones(num_features, bool)
        v_remain = np.ones(num_vertices, bool)
        f_density = np.zeros(num_features)
        
        community_sizes = np.fromiter((c.num_vertices for c in communities), int)
        
        num_edges_remain = e_remain.sum()
        num_feat_remain = f_remain.sum()
        comm_candidates = []
        while num_edges_remain and num_feat_remain:
            # Remove redundant features: Those which do not further limit the current validity
            f_is_redund_bool = np.all(self._validities[np.ix_(v_remain,f_remain)],0)
            #print(f"removng: {is_feat_redund} {map_feat[:]} {map_feat(None)}")
            f_is_redund = index_bool2int(f_remain)[f_is_redund_bool]
            f_remain[f_is_redund] = False
            
            #print(f"removed: {is_feat_redund} {map_feat[:]} {map_feat(None)}")
            if log.rlim.debug:
                log.debug(f'Removed {len(f_is_redund)} fetures: {f_is_redund}.')
            num_feat_remain = f_remain.sum()
            if num_feat_remain == 0:
                break
    
            f_density_rem = self._get_feature_densities(index_bool2int(f_remain), v_remain, e_remain)
            f_density[f_remain] = f_density_rem
            
            f_idx_best = np.nanargmax(f_density * f_remain)
            cur_dens = f_density[f_idx_best]*f_remain[f_idx_best]
            f_remain[f_idx_best] = False
            num_feat_remain -= 1
            
            comm_candidates.append(CommunityCandidate(density=float(cur_dens), feature=int(f_idx_best)))
            
            cur_validity = self._validities[:,f_idx_best]
            v_remain &= cur_validity
            cur_num_vertices = v_remain.sum()
            if cur_dens == 0:
                break
            
            num_edges_remaining_greedy = e_remain.sum()
            if communities:
                e_idx_induced = self.get_induced_edges(v_remain)
                e_comm_size_exceeded = community_sizes[e_comm_id[e_idx_induced]] > cur_num_vertices
                e_remain[e_idx_induced[e_comm_size_exceeded]] = True
                num_edges_remain = e_remain.sum()
            else:
                num_edges_remain = num_edges_remaining_greedy
            if log.rlim.progress:
                log.progress((f'Added feat: {f_idx_best:4} dens: {cur_dens:6.4} (add/del/tot:{len(comm_candidates)}/{num_feat_remain}/{num_features}) '
                              f'cover {cur_num_vertices}/{num_vertices} ({(cur_num_vertices/num_vertices)*100:5.1f}%). '
                              f'Stolen {num_edges_remain-num_edges_remaining_greedy} edges.'))
        
        candidate_densities = np.fromiter((c.density for c in comm_candidates),float)
        idx_best_cand = np.nanargmax(candidate_densities)
        best_density = candidate_densities[idx_best_cand]
        best_features = np.fromiter((c.feature for c in comm_candidates[:idx_best_cand+1]), int)
        v_densest_bool = np.any(self._validities[:,best_features], axis=1)
        e_densest = self.get_induced_edges(v_densest_bool)
        v_densest = np.where(v_densest_bool)[0]
        community = self.Community(
            density_remain=best_density, density_full=len(e_densest)/len(v_densest),
            vertices=v_densest, edges=e_densest, features=best_features)
        return community
    
    def run(self):
        g = self._g
        e_covered = np.zeros(g.num_edges(), bool)
        e_comm_id = np.zeros(g.num_edges(),int)
        communities = self.CommunityCollection()
        for cur_comm_id in range(self.k):
            community = self.residual_densest(e_covered, e_comm_id, communities)
            e_comm_id[community.edges] = cur_comm_id
            e_covered[community.edges] = True
            communities.append(community)
            log.info(f'Added community {cur_comm_id+1}/{self.k}: {community}')
            if e_covered.sum()==len(e_covered):
                break
        self._communities = communities    
        return communities
    
    def summary_dict(self, options):
        fields = ['k','communities']
        summary = self.summary_from_fields(fields)
        summary['subgroups'] = Summarisable.List(self.subgroups())
        return summary

if __name__ == '__main__':
    from sdcore.language import ConjunctionLanguage
    from sdcore.datasets import DatasetFactory    
    from tests.datasets import Datasets
    
#    ch = logging.FileHandler(file)
#    ch.setLevel(log.level)
#    ch.setFormatter(logging.Formatter(self.values.log_fmt))
#    log.addHandler(ch)
#    log.info(f'Added logging handler to "{file}".')
    log.setLevel(log.levels.PROGRESS)
    log.setLevel(log.levels.DEBUG)
    log.add_stderr()
    log.setFormatter()
    dsf= DatasetFactory()
    gd = Datasets.make_motivation()
    gd = dsf.load_dataset('amazon:movies,bool')
    gg = gd.asgraph(attributes=())
    lang = ConjunctionLanguage(gd)
    validities = np.stack(tuple(p.validate() for p in lang.predicates),1)
    v_comm, e_comm_id = GalbrunCommunities(gg, validities).run(5)
    print(v_comm)
    
    
    
    
    