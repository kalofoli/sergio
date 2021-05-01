'''
Created on May 1, 2021

@author: janis
'''

import numpy as np

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