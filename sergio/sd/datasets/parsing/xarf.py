'''
Created on Feb 5, 2021

@author: janis
'''

from . import parsers as par
from . import formatters as fmt
from types import SimpleNamespace
import re
from typing import NamedTuple, Any
import pandas
import numpy

class Properties(SimpleNamespace):
    def __contains__(self, attr): return hasattr(self, attr)

    

class DataBundle:
    class Relations(SimpleNamespace): pass
    class Matrices(SimpleNamespace): pass
    class Structures(SimpleNamespace): pass
    def __init__(self, relations={}, matrices={}, structures={}, name=''):
        self._name = name
        self.rels = self.Relations(**relations)
        self.structs = self.Structures(**structures)
        self.mats = matrices

class DataBundleFormatter:    
    def __generate__(self, fid):
        rs = Relation()
        for name,rel in self.rels.__dict__.items():
            rs.write(fid, rel, name=name)
    
    @classmethod
    def load(cls, fid, name):
        rs = Relation()
        rels = {}
        structs={}
        try:
            rid = ParserReader(fid)
            while True:
                rel = rid.__parse__(rs)
                if rel is None:
                    break
                rels.update({rel.name:rel})
        except ParseEOF:
            pass
        return cls(relations=rels, structures=structs, name=name)


class Graph:
    def __init__(self, edges, directed, vp=None, ep=None, num_vertices=None, name=None):
        self.edges = numpy.array(edges)
        self.vp = vp
        self.ep = ep
        self.directed = directed
        if num_vertices is None:
            if vp is None:
                num_vertices = self.edges.ravel().max()+1
            else:
                num_vertices = len(self.vp.index)
        self.num_vertices = num_vertices
        self.name = name
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
        

from .formatters import props

class DataManager():
    class Entry(NamedTuple):
        name: str
        symbol: fmt.Symbol
        data: Any
    def __init__(self):
        self._data = {}
    def add(self, name, symbol, data):
        self._data[name] = self.Entry(name=name, symbol=symbol, data=data)
    def __call__(self):
        for e in self._data.values():
            yield from e.symbol.__generate__(props(data=e.data, name=e.name))
    

class GraphFormatter(fmt.Symbol):
    from .formatters import LineChain, Header, Name, Token, Property, Relation, Matrix, DataSource
    SYM_HEADER = LineChain((Header('structure'), Name(), Token('graph')))
    SYM_GRAPH = LineChain((Header('graph'), Property()))
    SYM_VP = LineChain((Header('vertex-properties'), DataSource(Relation(), name='vp')))
    SYM_EP = LineChain((Header('edge-properties'), DataSource(Relation(), name='ep')))
    SYM_EDGES = LineChain((Header('edges'), DataSource(Matrix(by_columns=True),name='edges')))
    SYM_END = LineChain((Header('end'), Token('structure'), Name()))
    
    def __generate__(self, graph):
        yield from self.SYM_HEADER.__generate__(props(name=graph.name))
        yield from self.SYM_GRAPH.__generate__(props(directed=graph.directed))
        
        dm = DataManager()
        yield from self.SYM_EDGES.__generate__(props(data=graph.edges, manager=dm, name='edges'))
        if graph.ep is not None:
            yield from self.SYM_EP.__generate__(props(data=graph.ep, manager=dm, name='ep'))
        if graph.vp is not None:
            yield from self.SYM_VP.__generate__(props(data=graph.vp, manager=dm, name='vp'))
        
        yield from dm()
        
        yield from self.SYM_END.__generate__(props(name=graph.name))
        

class GraphParser(par.DataSymbol):
    from .parsers import LineChain, Header, Name, Token, Properties, Relation, Matrix, DataSource
    SYM_HEADER = LineChain((Header('structure'), Name(), Token('graph')))
    SYM_GRAPH = LineChain((Header('graph'), Properties()))
    SYM_VP = LineChain((Header('vertex-properties'), DataSource(Relation(), name='vp')))
    SYM_EP = LineChain((Header('edge-properties'), DataSource(Relation(), name='ep')))
    SYM_EDGES = LineChain((Header('edges'), DataSource(Matrix(by_columns=True),name='edges')))
    SYM_END = LineChain((Header('end'), Token('structure'), Name()))
    
    
    
    def _parse(self, rid):
        name = rid.parse(self.SYM_HEADER)
        directed = rid.parse(self.SYM_GRAPH)
        
        '''
        dm = DataManager()
        self.SYM_EDGES.__generate__(props(data=graph.edges, manager=dm, name='edges'))
        if graph.ep is not None:
            yield from self.SYM_EP.__generate__(props(data=graph.ep, manager=dm, name='ep'))
        if graph.vp is not None:
            yield from self.SYM_VP.__generate__(props(data=graph.vp, manager=dm, name='vp'))
        
        yield from dm()
        
        yield from self.SYM_END.__generate__(props(name=graph.name))
        '''
    
    
