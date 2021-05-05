'''
Created on May 3, 2021

@author: janis
'''
from colito.factory import FactoryBase, factorymethod, FactoryGetter,\
    FactoryDescription
from colito.summaries import SummarisableAsDict, SummaryOptions,\
    SummarisableList, SummarisableDict
import time

from .base import SearchState, SearchVisitor
from typing import NamedTuple, Dict
from colito.statistics import StatisticsBase

class ScoringFunctions(FactoryBase):
    
    @factorymethod(name='value_addition')
    @classmethod
    def make_value_addition(cls, mult_oest:float=1., mult_fval:float=1.):

        def score(state:SearchState) -> float:
            return (state.optimistic_estimate * mult_oest if mult_oest else 0) + (state.objective_value * mult_fval if mult_fval else 0)

        return score

    @factorymethod(name='value_multiplication')
    @classmethod
    def make_value_multiplication(cls, power_oest:float=1., power_fval:float=1.):

        def score(state:SearchState) -> float:
            return state.optimistic_estimate ** power_oest * state.objective_value ** power_fval

        return score
    
    @factorymethod(name='coverage')
    @classmethod
    def make_coverage(cls, increasing:bool=True):
        mult = 1 if increasing else -1

        def score(state:SearchState) -> float:
            nonlocal mult
            return mult * state.selector.validity.mean()
        
        return score

    @factorymethod(name='selector')
    @classmethod
    def make_selector(cls, record:str='indices'):
        def score(state:SearchState) -> float:
            nonlocal record
            return getattr(state.selector, record)
        
        return score
    
    @factorymethod(name='optimistic_estimate')
    @classmethod
    def make_optimistic_estimate(cls, increasing:bool=True):
        mult = 1 if increasing else -1

        def score(state:SearchState) -> float:
            nonlocal mult
            return mult * state.optimistic_estimate
        
        return score
    
    @factorymethod(name='hash')
    @classmethod
    def make_hash(cls, salt:str = '', volatile:bool=False):
        
        def score_non_volatile(state:SearchState) -> float:
            nonlocal salt
            return hash((state.selector,salt))
        def score_volatile(state:SearchState) -> float:
            nonlocal salt
            return hash((state.selector,state.depth,tuple(state.blacklist), tuple(state.covered), state.optimistic_estimate, salt))
        
        return score_volatile if volatile else score_non_volatile
    
    get_score = FactoryGetter(member='product', default='value_addition', classmethod=True)
    get = FactoryGetter(default='value_addition', classmethod=True)
    
    description = FactoryDescription()


class SearchResultLogger(SearchVisitor):
    class Update(NamedTuple):
        new: SearchState
        old: SearchState
        statistics: StatisticsBase
        time: float
    class UpdateList(list, SummarisableAsDict):
        def __summary_dict__(self, options:SummaryOptions):
            selector2index = {}
            update_dicts = SummarisableList()
            for update in self:
                if update.new.selector not in selector2index:
                    selector2index[update.new.selector] = len(selector2index)
                state_new = update.new.__summary_dict__(selector_dict=selector2index)
                index_old = -1 if update.old is None else selector2index[update.old.selector]
                update_dict = SummarisableDict(state_new)
                update_dict['index_old'] = index_old
                update_dict['statistics'] = update.statistics
                update_dict['time'] = update.time
                update_dicts.append(update_dict)
            dct = {'selectors': SummarisableList(selector2index.keys()),
                   'updates':update_dicts
                   }
            return dct
    def __init__(self, results=None):
        if results is None:
            results = []
        self._result_history = self.UpdateList(results)
    def result_added(self, state:SearchState, result_old:'LanguageTopKBranchAndBound.Result'):
        update = self.Update(new=state, old=result_old, statistics=state.search.statistics.copy(), time=time.time())
        self._result_history.append(update)
    result_history = property(lambda self:self._result_history, None, 'Results tracked so far.')




class GraphingSearchVisitor():
    
    def __init__(self):
        self._graph = None
        self._vertices:Dict[SearchState,int] = {}
        self.reset()
        
        
    def reset(self):
        from graph_tool import Graph
        graph = Graph(directed=True)
        graph.vp['valid'] = graph.new_vertex_property('bool')
        graph.ep['order'] = graph.new_edge_property('int')
        graph.vp['name'] = graph.new_vertex_property('string')
        graph.vp['value'] = graph.new_vertex_property('float')
        self._graph = graph
        self._vertices = {}

    @property
    def graph(self):
        return self._graph
    
    def start(self, dfs, root_state):
        self.reset()
        
    def stop(self, dfs):
        pass
    
    def state_popped(self, dfs, state):
        pass
    
    def state_expanded(self, dfs, state, valid_states, new_states):
        valid_selectors = set(s.selector for s in valid_states)
        for new_state in new_states:
            pruned = new_state.selector not in valid_selectors
            self.add_edge(state, new_state, pruned=pruned)
            
    def result_added(self, state, result_old): 
        #print(f'New result: {state}')
        pass

    def get_vertex(self, state):
        if state.selector not in self._vertices:
            graph = self._graph
            vertex = graph.add_vertex()
            self._vertices[state.selector] = vertex 
            graph.vp.valid[vertex] = True
            graph.vp['name'][vertex] = str(state.selector.indices_path)             
            graph.vp['value'][vertex] = state.objective_value             
        return self._vertices[state.selector]
        
            
    def add_edge(self, state_src, state_dst, pruned): # dbg_remove
        vertex_dst = self.get_vertex(state_dst)
        vertex_src = self.get_vertex(state_src)
        #print(vertex_src, vertex_dst)
        graph = self._graph
        edge = graph.add_edge(vertex_src,vertex_dst)
        graph.vp['valid'][vertex_dst] = not pruned
        graph.ep['order'][edge] = graph.num_edges()
        #graph.set_vertex_filter(graph.vp.valid)

    def draw(self, **kwargs):
        from matplotlib import cm
        from graph_tool.draw import graph_draw
        
        graph = self._graph
        draw_args = {'vertex_text':graph.vp['name'],
                    'edge_color':graph.ep['order'],
                    'vertex_fill_color':graph.vp.value,
                    'vertex_font_size':9,
                    'vcmap':cm.hot,
                    'ecmap':cm.winter}
        draw_args.update(kwargs)
        graph_draw(self.graph,**draw_args)
