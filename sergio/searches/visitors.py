'''
Created on Jul 17, 2018

@author: janis
'''

from sdcore.searches import DFSVisitor
import graph_tool as gt
from graph_tool import draw
from matplotlib import cm

class GraphingDFSVisitor(DFSVisitor):
    
    def __init__(self):
        self._graph = gt.Graph(directed=True)
        self._vertices:Dict[DFSState,int] = {}
        self.reset()
        
        
    def reset(self):
        graph = gt.Graph(directed=True)
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
            
    def result_added(self, state): pass

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
        graph = self._graph
        edge = graph.add_edge(vertex_src,vertex_dst)
        graph.vp['valid'][vertex_dst] = not pruned
        graph.ep['order'][edge] = graph.num_edges()
        graph.set_vertex_filter(graph.vp.valid)

    def draw(self, **kwargs):
        graph = self._graph
        draw_args = {'vertex_text':graph.vp['name'],
                    'edge_color':graph.ep['order'],
                    'vertex_fill_color':graph.vp.value,
                    'vertex_font_size':9,
                    'vcmap':cm.hot,
                    'ecmap':cm.winter}
        draw_args.update(kwargs)
        draw.graph_draw(self.graph,**draw_args)
