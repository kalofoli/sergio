'''
Created on Feb 3, 2021

@author: janis
'''
from types import SimpleNamespace
from sergio.sd.datasets.parsing.parsers import RewindableReader, ParserReader


class DataBundle:
    class Relations(SimpleNamespace): pass
    class Structures(SimpleNamespace): pass
    def __init__(self, relations, structures, name):
        self._name = name
        self.rels = self.Relations(**relations)
        self.structs = self.Structures(**structures)
    
    def write(self, fid):
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

if __name__ == '__main__':
    import doctest
    from parsing import parsers
    doctest.testmod(parsers)

 
if __name__ == '__main_':
    
    import pandas as pd
    
    import numpy as np
    import scipy as sp
    import scipy.sparse
    import sys
    from parsing.xarf import Graph, GraphFormatter

    if False:
        edges = np.c_[[0,0,0,1,1,2,3,4,3,4,4,5,5,6,6,7,8],[1,2,3,2,3,8,8,7,6,9,10,10,9,8,9,8,10]]
        vp = pd.DataFrame({'label':np.r_[0,0,0,0,0,1,1,1,1,1],'weight':np.r_[.1,.2,.1,.2,.4,.5,.6,.5,.6,.9]})
        ep = pd.DataFrame({'weight':[0,0,0,1,1,1,2,2,2,3,3,3,3,3,4,4,4]})
        
        g = Graph(edges = edges, vp=vp, ep=ep, directed=True, num_vertices=10, name='g1')
        s = GraphFormatter().dump(sys.stderr,g)
        print(s)
        
        
        
        
        
        
        sys.exit()
        
    
    d = '''@structure g1 graph
@graph directed=True
@edges edges
@edge-properties ep
@vertex-properties vp
@matrix edges
@data type=dense order=columns
0,0,0,1,1,2,3,4,3,4,4,5,5,6,6,7,8
1,2,3,2,3,8,8,7,6,9,10,10,9,8,9,8,10
@end data
@end matrix edges
@relation ep
@columns weight
@attribute 1 numeric storage=int
@data type=dense order=columns
0,0,0,1,1,1,2,2,2,3,3,3,3,3,4,4,4
@end data
@end relation ep
@relation vp
@columns label weight
@attribute 2 numeric storage=float
@attribute 1 numeric storage=int
@data type=dense order=columns
0,0,0,0,0,1,1,1,1,1
0.1,0.2,0.1,0.2,0.4,0.5,0.6,0.5,0.6,0.9
@end data
@end relation vp
@end structure g1
'''
    from io import StringIO
    s='''@matrix edges
@data type=dense order=columns
0,0,0,1,1,2,3,4,3,4,4,5,5,6,6,7,8
1,2,3,2,3,8,8,7,6,9,10,10,9,8,9,8,10
@end data
@end matrix edges
'''
    t = StringIO(d)
    from parsing.xarf import GraphParser
    rid = ParserReader(t)
    p = rid.parse(GraphParser())
    print(p)
    
    sys.exit()
    
    db = DataBundle.load(t, 'test')
    M = sp.sparse.csc_matrix(([1,2,1,1,2,2,6],[[1,2,2,2,3,3,4],[1,1,1,2,2,3,5]]))
    df_a = pd.DataFrame.sparse.from_spmatrix(M)
    M = sp.sparse.csc_matrix(([4,4,3,1,2,5,3],[[1,1,1,2,6,6,7],[1,1,2,4,4,5,5]]))
    df_b = pd.DataFrame.sparse.from_spmatrix(M)
    db = DataBundle({'A':df_a, 'B':df_b},{}, 'ego-twitter')
    import sys
    g = db.write(sys.stderr)
    print(g)
