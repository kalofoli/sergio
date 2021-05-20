'''
Created on Mar 2, 2021

@author: janis
'''

from typing import List
import pandas as pd
import numpy as np


from sergio.data.bundles.entities import EntityAttributes
from sergio.attributes import AttributeKind



class EntityStructure:
   # __slots__ = ()
   pass

class EntityAttributesWithStructures(EntityAttributes):
    __target_name__ = 'structures'
    def __init__(self, structures:List[EntityStructure], attribute_data: pd.DataFrame, name: str='unknown',
                 attribute_info={None:None}, attribute_selection={None:1}) -> None:
        super().__init__(attribute_data = attribute_data, name = name,
                 attribute_info=attribute_info, attribute_selection=attribute_selection)

        self._structures = structures

    @property
    def structures(self): return self._structures
    @property
    def target_name(self): return self.__target_name__
    target_data = structures

class EntityAttributesWithGraphs(EntityAttributesWithStructures):
    '''
    >>> from sergio.data.factory.morris import MorrisLoader
    >>> ml = MorrisLoader(data_home='datasets/morris')
    >>> ds = ml.load_dataset('MUTAG')
    >>> ea = EntityAttributesWithGraphs.from_morris(**ds.__dict__)
    >>> ea
    <EntityAttributesWithGraphs[MUTAG](188x1/1)>
    >>> eas = ea.slice[10:20]; eas
    <EntityAttributesWithGraphs[MUTAG](10x1/2)>
    >>> eass = eas.slice[2:6]; eass
    <EntityAttributesWithGraphs[MUTAG](4x1/2)>
    >>> np.all(ea.attribute_data.iloc[eas.attribute_data.orig_idx].target.values == eas.attribute_data.target.values)
    True
    >>> [ea.target_data[i] for i in eas.attribute_data.orig_idx] == eas.target_data
    True
    >>> np.all(ea.attribute_data.iloc[eass.attribute_data.orig_idx].target.values == eass.attribute_data.target.values)
    True
    >>> [ea.target_data[i] for i in eass.attribute_data.orig_idx] == eass.target_data
    True
    '''
    __target_name__ = 'graphs'

    @classmethod
    def from_morris(cls, name, data, target, with_oidx:bool=False, directed:bool=False):
        '''
        :param with_oidx: Append to the edge properties (if existing) the original edge index.
        :param directed: Specify if the graphs should be considered directed. If None, they are directed if reverse edges are detected.
        '''
        structures = [Graph.from_morris(*entry, with_orig_idx=with_oidx, directed=directed)
                      for entry in data]
        df_attr = pd.DataFrame({'target':target})
        ea = cls(structures=structures, name=name, attribute_data = df_attr)
        return ea

    class _Slicer:
        def __init__(self, ea):
            self._ea = ea
        def __getitem__(self, what):
            ea = self._ea
            df_attrs = ea.attribute_data.iloc[what].copy()
            attr_info = ea.attribute_info
            attr_sel = ea.attribute_selection
            idx = df_attrs.index

            if 'orig_idx' not in df_attrs.columns:
                df_attrs['orig_idx'] = idx
                attr_info = attr_info + [AttributeKind.INDEX]
                attr_sel = np.r_[attr_sel, False]
            df_attrs = df_attrs.reset_index(drop=True)
            structures = [ea.structures[i] for i in idx]
            ea_sl = ea.__class__(structures=structures, attribute_data=df_attrs, name=ea.name,
                                 attribute_info=attr_info, attribute_selection=attr_sel)
            return ea_sl
    @property
    def slice(self): return self._Slicer(self)


class Graph(EntityStructure):
    #__slots__ = ('edges','vp','ep','directed', 'num_vertices')
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
    @classmethod
    def from_morris(cls, edges, vp=None, ep=None, directed=None, with_orig_idx=True):
        import numpy as np
        E = np.vstack(tuple(edges))
        df_edges = pd.DataFrame(E, columns=tuple('sd'))\
            .assign(increasing=E[:,0]>E[:,1], same=E[:,0]==E[:,1], orig_idx=np.arange(E.shape[0]))\
            .set_index(['s','d']).sort_index().assign(idx=np.arange(E.shape[0]))
        if vp is not None:
            df_vp = pd.DataFrame({'label':vp.values()}, index=np.fromiter(vp, int)-1)
        else:
            df_vp = None
        if ep is not None:
            ep_idx = df_edges.loc[list(ep)].idx
            df_ep = pd.DataFrame({'label':ep.values()}, index=ep_idx.values).sort_index()
            if with_orig_idx:
                df_ep['orig_idx'] = pd.Series(df_edges.orig_idx.values, index=df_edges.idx)
            if directed is None:
                num_inc = df_edges.increasing.sum()
                num_dec = len(df_edges)-df_edges.same.sum()
                directed = not (num_inc==0 or num_dec==0)
        else:
            if directed is None:
                directed = False
            df_ep = None
        E = np.vstack(df_edges.index)-1
        g = cls(edges=E, directed=directed, vp = df_vp, ep = df_ep)
        return g
    def to_igraph(self, vp=None, ep=None):
        import igraph
        g = igraph.Graph(self.edges)
        if vp is None:
            vp = self.vp.columns if self.vp is not None else []
        if ep is None:
            ep = self.ep.columns if self.ep is not None else []
        for vprop in vp:
            g.vs[vprop] = self.vp[vprop].values
        for eprop in ep:
            g.es[eprop] = self.ep[eprop].values
        return g


if __name__ == '__main__':
    import doctest
    doctest.testmod()
