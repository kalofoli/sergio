'''
Created on May 1, 2021

@author: janis
'''

import numpy as np
import pandas as pd
from sergio.data.bundles import EntityAttributesWithStructures
import re

from colito.logging import getModuleLogger

log = getModuleLogger(__name__)


class SQLKernelStore():
    def __init__(self, db):
        self._db = db
        self._init()
        
    def _init(self):
        self._db.execute('''create table if not exists drug_kernel (
            tag varchar, score_type varchar, data blob,
            created date,
            primary key (tag, score_type)
        );''') 

    @staticmethod
    def pack_byte_kernel(K, cids):
        import struct
        from io import BytesIO
        n = len(cids)
        b = BytesIO()
        b.write(struct.pack('>I',n))
        b.write(struct.pack(f'<{n}Q', *cids))
        data = K[np.triu_indices_from(K)]
        fmt = f'{int(n*(n+1)/2)}B'
        b.write(struct.pack(fmt, *data))
        return bytes(b.getbuffer())
    @staticmethod
    def unpack_byte_kernel(data):
        import struct
        from io import BytesIO
        read = lambda fmt: struct.unpack(fmt, b.read(struct.calcsize(fmt)))
        b = BytesIO(data)
        n = read('>I')[0]
        cids = np.array(read(f'<{n}Q'))
        data = np.array(read(f'{int(n*(n+1)/2)}B'))
        K = np.zeros((n,n), np.uint8)
        K[np.triu_indices_from(K)] = data
        K = K + np.triu(K,1).T
        return K,cids
    def store(self, tag, score_type, K, cids):
        import datetime
        blob = self.pack_byte_kernel(K, cids)
        args = {'tag':tag, 'created':datetime.datetime.now(), 'data':blob, 'score':score_type}
        self._db.execute('insert into drug_kernel (tag, score_type, created, data) values(:tag, :score,:created, :data)', args)
        self._db.commit()
    def load(self, tag, score, qcids1, qcids2=None):
        '''Fetch the query cids from a stored kernel.
        
        '''
        cur = self._db.execute('select data from drug_kernel where tag=:tag and score_type=:score', {'tag':tag, 'score':score})
        rows = cur.fetchone()
        if not rows:
            raise ValueError('No such row')
        blob = rows[0]
        K_all, cids_all = self.unpack_byte_kernel(blob)
        from colito.indexing import Indexer
        indexer = Indexer(cid=cids_all)
        idx_qcids1 = indexer.cid2indices[qcids1].values
        if qcids2 is None:
            idx_qcids2 = idx_qcids1
        else:
            idx_qcids2 = indexer.cid2indices[qcids2].values
        K = K_all[np.ix_(idx_qcids1, idx_qcids2)]
        return K

    def list(self):
        cur = self._db.execute('select tag, score_type from drug_kernel;')
        return cur.fetchall()
    
    
    
class Chembl:
    def __init__(self, file):
        import sqlite3
        self._db = sqlite3.connect(file)
        self._tables = [row[0] for row in 
                        self._db.execute('select name from SQLITE_MASTER where type="table" and name glob "drug*";').fetchall()]
        
    @property
    def db(self): return self._db
    @property
    def parts(self): return [tbl.split('_')[1] for tbl in set(self._tables)]
    @property
    def attribute_parts(self): return list(set(self.parts) - {'xref','kernel','structs'})
    @property
    def kernels(self): return self._db.execute('select tag, score_type from drug_kernel;').fetchall()
    
    def get_part(self, part):
        return pd.read_sql_query(f'select * from drug_{part};', self._db)

    def assemble(self, parts, merge='inner'):
        df = self.get_part(parts[0])
        for part in parts[1:]:
            df_part = self.get_part(part)
            df = pd.merge(df, df_part, on='molregno', how=merge)

        return df

    def get_pubchem_scores_from_df(self, df_all, kernel):
        df_xref = self.get_part('xref')

        df = pd.merge(df_all, df_xref.loc[:,['molregno','pubchem']], on='molregno', how='inner')

        cids = df.pubchem.values
        K = self.get_pubchem_scores(cids, kernel=kernel)
        return df, K, cids
        
    def get_pubchem_scores(self, cids, kernel):

        ks = SQLKernelStore(self.db)
        K = ks.load(*kernel, qcids1=cids)/100
        return K

    NORMALISE_COLUMN_EXCLUDE = {'mec_ids'}
    NORMALISE_COLUMN_INDICES = {'pubchem','molregno'}
    NORMALISE_COLUMN_BINARISE = re.compile('^(L[1-3]|act|tar|org|efo)_')
    def normalise(self, df, tag, binarise=True, exclude=True, keep_indices=True, kernel=None):
        if kernel is not None:
            df_out, K, cids = self.get_pubchem_scores_from_df(df, kernel=kernel)
            stag = 'kr[' + ':'.join(kernel) + ']'
        else:
            df_out = df.copy()
            stag = 'krN'
        selected = np.ones(df_out.shape[1], bool)
        kinds = np.array([None]*df_out.shape[1])
        is_bool = df_out.columns.str.match(self.NORMALISE_COLUMN_BINARISE)
        if binarise:
            df_out.loc[:,is_bool] = df_out.loc[:,is_bool].astype(bool).astype(int)
            kinds[is_bool] = 'boolean'
            stag = f'{stag}BIN'
        else:
            stag = f'{stag}NUM'
            kinds[is_bool] = 'NUMERICAL'
        is_excl = np.array([col in self.NORMALISE_COLUMN_EXCLUDE for col in df_out.columns],bool)
        if exclude:
            df_out = df_out.loc[:,~is_excl]
            kinds = kinds[~is_excl]
            selected = selected[~is_excl]
            stag = f'{stag}exY'
        else:
            stag = f'{stag}exN'
            selected[is_excl] = False
        if kernel is None:
            structs = df_out.index.values
            for index in self.NORMALISE_COLUMN_INDICES:
                if index in df_out.columns:
                    log.info(f'Using as struct index column: {index}')
                    structs = df_out[index]
                    break
        else:
            log.info(f'Using kernel (pubchem) cids as struct indices')
            structs = cids
        is_index = np.array([col in self.NORMALISE_COLUMN_INDICES for col in df_out.columns],bool)
        if not keep_indices:
            df_out = df_out.loc[:,~is_index]
            kinds = kinds[~is_index]
            selected = selected[~is_index]
            stag = f'{stag}idY'
        else:
            stag = f'{stag}idN'
            selected[is_index] = False
            kinds[is_index] = 'INDEX'
        from sergio.data.bundles import EntityAttributesWithStructures
        
        name = f'chembl-drugs-{stag}-{tag}'
        eas = EntityAttributesWithStructures(
            attribute_data=df_out,attribute_info=kinds, attribute_selection=selected,
            structures=structs,
            name=name
        )
        if kernel is not None:
            return eas, K, cids
        else:
            return eas
    def make_data(self, parts=None, merge='inner', binarise=True, exclude=True, keep_indices=True, kernel=None):
        if parts is None:
            parts = self.attribute_parts
        df = self.assemble(parts=parts, merge=merge)
        sparts = ''.join(f'{part[0].upper()}{part[1:].lower()}' for part in parts)
        tag = f'{merge.lower()}-{sparts}'
        return self.normalise(df, binarise=binarise, exclude=exclude, keep_indices=keep_indices, kernel=kernel, tag=tag)

if __name__ == '__main__':
    chembl = Chembl('data/chembl-drugs.sqlite3')
    
    eas, K, cids = chembl.make_data(binarise=True, exclude=False, keep_indices=True, kernel=chembl.kernels[0])
    
    _K = chembl.get_pubchem_scores(eas.structures, chembl.kernels[0])
    assert np.all(K == _K),'Failed to reconstruct kernel'