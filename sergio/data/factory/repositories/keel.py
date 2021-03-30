'''
Created on Feb 1, 2021

@author: janis
'''

import re
from collections import namedtuple
import pandas

from colito.logging import getLogger
from sergio.data.factory.repositories.loader import RepositoryLoader
from sergio.data.factory import read_url
import os
from sergio.data.bundles.entities import EntityAttributesWithAttributeTarget

log = getLogger(__name__)

DatasetEntry = namedtuple('DatasetEntry',('name','reals','integers','nominals','rows','incomplete_rows','classes', 'has_missing','url','kind'))


class KeelLoader(RepositoryLoader):
    """
    >>> kl = KeelLoader('datasets/keel')
    >>> kl.load_metadata().shape[1]
    9
    >>> kl.load_dataset('german')
    <EntityAttributesWithAttributeTarget[german](999x20/21) target: Customer(category@20)>
    """
    __name__ = 'keel'
    _KIND_MAP = {'regression':'reg','classification':'clas'}
    URL_META = 'https://sci2s.ugr.es/keel/category.php?cat={tag}'
    URL_BASE = 'https://sci2s.ugr.es/keel/'
    
    _DatasetEntry = namedtuple('DatasetEntry',('name','reals','integers','nominals','rows','incomplete_rows','classes', 'has_missing','url','kind'))

    rex_attrs = re.compile('\s*\((?P<r>[0-9]+)/(?P<i>[0-9]+)/(?P<n>[0-9]+)\s*')
    rex_rows = re.compile('\s*(?P<all>[0-9]+)\s+\((?P<miss>[0-9]+)\)\s*')
    
    def get_dataset_url(self, name):
        tail = self.metadata.loc[name].url
        url = os.path.join(self.URL_BASE, tail)
        return url
        
    def parse_dataset_data(self, name, path):
        file = f'{name}.dat'
        
        with path.open(file, 'r') as fid:
            ea = self.read_xarf(fid)
        return ea

    @classmethod
    def parse_entry(cls, e, kind):
        if kind == 'classification':
            with_class = True
        elif kind == 'regression':
            with_class = False
        else:
            raise NotImplementedError(f'Unsupported kind {kind}')
        name = e.xpath('.//td[@class="dataD"][1]/a/text()')[0]
        atrs = e.xpath('.//td[@class="dataDAT"]/span/following-sibling::text()')[0]
        num_reals, num_ints, num_noms = map(int, cls.rex_attrs.match(atrs).groups())
        snum_rows = e.xpath('.//td[@class="dataD"][2]/text()')[0]
        if with_class:
            num_classes = int(e.xpath('.//td[@class="dataD"][3]/text()')[0])
            offset = 1
        else:
            num_classes = 0
            offset = 0
        has_missing = e.xpath(f'.//td[@class="dataD"][{3+offset}]/text()')[0] == 'Yes'
        if has_missing:
            num_full, num_rows = map(int, cls.rex_rows.match(snum_rows).groups())
            num_mising = num_rows-num_full
        else:
            num_rows = int(snum_rows)
            num_mising = 0
        url = e.xpath(f'.//td[@class="dataD"][{4+offset}]/a/@href')[0]
        return cls._DatasetEntry(name=name, reals=num_reals, integers=num_ints, nominals=num_noms, rows=num_rows, incomplete_rows=num_mising,classes=num_classes, has_missing=has_missing, url=url, kind=kind)
    
    @classmethod
    def _download_metadata(cls):
        from lxml import etree
        entries = []
        for kind,tag in cls._KIND_MAP.items():
            url = cls.URL_META.format(tag=tag)
            data = read_url(url)
            root = etree.HTML(data)
            entries += [cls.parse_entry(e, kind) for e in root.xpath("//table[@class='data']/tbody/tr[@class='rowD']/td[@class='dataDAT']/..")]
        df_datasets = pandas.DataFrame(entries).set_index('name')        
        return df_datasets


    rex_header = re.compile('^\s*@(?P<header>[a-zA-Z_]+)\s+(?P<rest>.*?)\s*$')
    rex_attr = re.compile('(?P<name>[^\s]+)\s+(?P<spec>.*)\s*')
    @classmethod
    def read_xarf(cls, file):
        selection = {None:True}
        kinds = {None:None}
        rel_name = None
        target = None
        dtypes = {}
        columns = []
        while True:
            line = file.readline().decode('latin')
            tag, rest = cls.rex_header.match(line).groups() 
            if tag == 'data':
                df = pandas.read_csv(file, header=0, names=columns, dtype=dtypes)
                break
            elif tag == 'relation':
                rel_name = rest
            elif tag == 'attribute':
                name, spec = cls.rex_attr.match(rest).groups()
                columns.append(name)
                if spec[0] == '{':
                    dtypes[name] = 'category'
                elif spec[0] == 'i':
                    dtypes[name] = 'int'
            elif tag == 'inputs':
                selection = {None:False, **{k:True for k in re.split(',\s*',rest)}}
            elif tag == 'outputs':
                target = rest
        
        ea = EntityAttributesWithAttributeTarget(
            name = rel_name,
            attribute_data=df,
            attribute_selection=selection,
            attribute_info=kinds,
            target=target
        )
        return ea
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    