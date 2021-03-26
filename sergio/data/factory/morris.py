'''
Created on Feb 1, 2021

@author: janis
'''
import os
import zipfile
import re
from . import read_url, save_url
from collections import Counter
from io import TextIOWrapper
import numpy
from types import SimpleNamespace as SNS
import pandas
from colito.logging import getLogger

log = getLogger(__name__)

class MorrisLoader():
    
    url = 'https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets'
    
    metadata = None
    
    def __init__(self, data_home='.'):
        self.data_home = data_home
    
    def list_datasets(self):
        if self.metadata is None:
            self.load_metadata()
        return list(self.metadata.index)

    @property
    def meta_file(self): return os.path.join(self.data_home,'datasets.csv')
    def load_dataset(self, name):
        meta = self.load_metadata()
        url = meta.loc[name].download
        with self.fetch_zip_cached(url, name) as zip_ref:
            log.info(f'Parsing dataset {name}')
            dataset = read_data(name,path=zip_ref)
        
        #dataset.data = SliceableList(dataset.data)
        dataset.name = name
        return dataset
    
    def fetch_zip_cached(self, url, name):
        file_name = os.path.join(self.data_home,f'{name}.zip')
        try:
            zip_fid = zipfile.ZipFile(file_name,'r')
        except (OSError,zipfile.BadZipFile) as exc:
            log.info(f'{exc} Downloading...')
            save_url(url, file_name)
            zip_fid = zipfile.ZipFile(file_name,'r')
        return zip_fid

    def load_metadata(self, cache=True, force_reload=False):
        df = None
        meta_file = self.meta_file
        def download():
            df = self._download_metadata(simplify=True)
            if cache and meta_file is not None:
                df.to_csv(meta_file, sep='\t')
            return df
        if force_reload:
            df = download()
        else:
            if self.metadata is not None:
                df = self.metadata
            else:
                log.info(f'Trying to load cache: {meta_file}')
                try:
                    df = pandas.read_csv(meta_file, sep='\t').set_index('name')
                except Exception as e:
                    log.info(f'Could not load cache: {meta_file} because of: {e}')
                    df = download()

        self.__class__.metadata = df
        return self.metadata
        
    @classmethod
    def _download_metadata(cls, simplify=True):
        from lxml import etree
    
        data = read_url(cls.url)
        root = etree.HTML(data)
    
        header_cats = []
        for e in root.xpath('//table/tr[1]/th'):
            colspan = int(e.attrib.get('colspan',1))
            txt = e.xpath('*/text()')[0]
            header_cats += [txt]*colspan
        header_sub_cats = []
        for e in root.xpath('//table/tr[2]/*'):
            txt = e.xpath('*/text()')
            header_sub_cats += txt if txt else [None]
        
    
        erows = root.xpath('//table/tr[position()>2 and count(td)>2]')
        rex_bool = re.compile('^\s*(?P<bool>[+–R])(\s\((?P<count>[0-9]+)\))?\s*$')
        def parse_elem(e):
            try:
                if e.text is None:
                    val = e.xpath('a/@href')[0]
                else:
                    txt = e.text
                    try:
                        val = int(txt)
                    except ValueError:
                        try:
                            val = float(txt)
                        except ValueError:
                            m = rex_bool.match(txt)
                            if m is not None:
                                count = m.group('count')
                                if count is not None:
                                    val = int(m.group('count'))
                                else:
                                    val = True if m.group('bool') == '+' else False
                            else:
                                val = txt
            except Exception as exc:
                print(f'{exc} {type(exc)} at {e.xpath("*/text()")}')
                val = None
            return val
        datasets = list(list(map(parse_elem, erow)) for erow in erows)
        df_prn = pandas.DataFrame(datasets,columns=pandas.MultiIndex.from_tuples(zip(header_cats,header_sub_cats)))
        if simplify:
            df = cls.simplify_meta(datasets,header_cats,header_sub_cats)
        else:
            df = df_prn
        return df
        
        

    @classmethod
    def simplify_meta(cls, datasets, header_cats,header_sub_cats):
        
        def substitute(rexes,*args):
            results = []
            for data in args:
                if isinstance(data, list):
                    result = substitute(rexes,*data)
                else:
                    if data is None:
                        result = None
                    else:
                        result = data
                        for rex in rexes:
                            result = re.sub(*rex,result)
                results.append(result)
            return results
        cols_cats,cols_sub_cats = substitute([
            ('\s*\([^(]+\)',''),
            ('\W+',' '),
            ('\s+of\s*',' '),
            ('\s*$',''),
            ('\s+','_'),
            ('(.*)',lambda x:str.lower(x.group(0)))
            ],header_cats,header_sub_cats)
        colnames = tuple(f'{s}' if s is not None else c for c,s in zip(cols_cats,cols_sub_cats))
        df = pandas.DataFrame(datasets,columns=colnames).set_index('name')\
            .assign(node_attr=lambda x:x.node_attr.astype(int))\
            .assign(edge_attr=lambda x:x.edge_attr.astype(int))\
            .assign(num_classes=lambda x:x.num_classes.astype(int))
        return df


def read_data(
        name,path='',
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=False, fopen=open):
    """Create a dataset iterable for GraphKernel.

    Parameters
    ----------
    name : str
        The dataset name.

    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.

    is_symmetric : bool, default=False
        Defines if the graph data describe a symmetric graph.

    Returns
    -------
    Gs : iterable
        An iterable of graphs consisting of a dictionary, node
        labels and edge labels for each graph.

    classes : np.array, case_of_appearance=with_classes==True
        An one dimensional array of graph classes aligned with the lines
        of the `Gs` iterable. Useful for classification.

    """
    
    if isinstance(path, zipfile.ZipFile):
        zip_ref = path
        class ZipOpen:
            def __init__(self, *args, **kwargs):
                self.fid = zip_ref.open(*args, **kwargs)
            def __iter__(self):
                return iter(TextIOWrapper(self.fid))
            def __enter__(self):
                self.tio = TextIOWrapper(self.fid)
                return self.tio.__enter__()
            def __exit__(self, exc_type, exc_val, exc_tb):
                return self.tio.__exit__(exc_type, exc_val, exc_tb)
        fopen = ZipOpen
        folder = ''
    else:
        fopen = open
        folder = path
    
    get_component_path = lambda cmp: os.path.join(folder, f'{name}', f'{name}_{cmp}.txt')
    indicator_path = get_component_path("graph_indicator")
    edges_path = get_component_path('A')
    node_labels_path = get_component_path('node_labels')
    node_attributes_path = get_component_path("node_attributes")
    edge_labels_path = get_component_path("edge_labels")
    edge_attributes_path = get_component_path("edge_attributes")
    graph_classes_path = get_component_path("graph_labels")

    # node graph correspondence
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()

    # Associate graphs nodes with indexes
    with fopen(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()

    # Extract graph edges
    with fopen(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
            if is_symmetric:
                Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

    # Extract node attributes
    has_attrs = False
    if prefer_attr_nodes:
        try:
            with fopen(node_attributes_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    node_labels[ngc[i]][i] = \
                        [float(num) for num in
                         line[:-1].replace(' ', '').split(",")]
            has_attrs = True
        except KeyError:
            pass
    # Extract node labels
    elif not has_attrs:
        try:
            with fopen(node_labels_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    node_labels[ngc[i]][i] = int(line[:-1])
            has_attrs = True
        except KeyError: pass
    elif not has_attrs and produce_labels_nodes:
        for i in range(1, len(Graphs)+1):
            node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))

    # Extract edge attributes
    has_attrs = False
    if prefer_attr_edges:
        try:
            with fopen(edge_attributes_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    attrs = [float(num)
                             for num in line[:-1].replace(' ', '').split(",")]
                    edge_labels[ngc[elc[i][0]]][elc[i]] = attrs
                    if is_symmetric:
                        edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = attrs
            has_attrs = True
        except KeyError: pass
    # Extract edge labels
    elif not has_attrs:
        try:
            with fopen(edge_labels_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    edge_labels[ngc[elc[i][0]]][elc[i]] = int(line[:-1])
                    if is_symmetric:
                        edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = \
                            int(line[:-1])
            has_attrs = True
        except KeyError: pass
    Gs = list()
    if as_graphs:
        for i in range(1, len(Graphs)+1):
            Gs.append(Graph(Graphs[i], node_labels[i], edge_labels[i]))
    else:
        for i in range(1, len(Graphs)+1):
            Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

    if with_classes:
        classes = []
        with fopen(graph_classes_path, "r") as f:
            for line in f:
                classes.append(int(line[:-1]))

        classes = numpy.array(classes, dtype=numpy.int)
        return SNS(data=Gs, target=classes)
    else:
        return SNS(data=Gs)

