'''
Created on Mar 2, 2021

@author: janis
'''

from typing import List
from pandas import DataFrame

from .entities import EntityAttributes



class EntityStructure: pass

class EntityAttributesWithStructures(EntityAttributes):
    
    def __init__(self, structures:List[EntityStructure], attribute_data: DataFrame, name: str='unknown',
                 attribute_info={None:None}, attribute_selection={None:1}) -> None:
        super().__init__(attribute_data = attribute_data, name = name,
                 attribute_info=attribute_info, attribute_selection=attribute_selection)
        
        self._structures = structures

    @property
    def structures(self): return self._structures
    
class EntityAttributesWithGraphs(EntityAttributesWithStructures): pass

class Graph(EntityStructure):
    def __init__(self, edges, directed, vp=None, ep=None, num_vertices=None):
        import numpy as np
        self.edges = np.array(edges)
        self.vp = vp
        self.ep = ep
        self.directed = directed
        if num_vertices is None:
            if vp is None:
                num_vertices = self.edges.ravel().max()+1
            else:
                num_vertices = len(self.vp.index)
        self.num_vertices = num_vertices
    @property
    def num_edges(self): return self.edges.shape[0]
    def __str__(self):
        s = []
        if self.vp is not None: s.append(f'vp:{list(self.vp.columns)}')
        if self.ep is not None: s.append(f'ep:{list(self.ep.columns)}')
        sprops = ' '+' and '.join(s) if s else ''
        return f'{"Directed" if self.directed else "Undirected"} w/ |V|={self.num_vertices} |E|={self.num_edges}{sprops}'
    def __repr__(self):
        return f'<{type(self).__name__}({self!s})>'


if __name__ == '__main__':
    import doctest
    doctest.testmod()