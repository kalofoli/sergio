'''
Created on Jan 14, 2020

@author: janis
'''

from collections import namedtuple
import heapq

import numpy as np
from typing import List, NamedTuple, Tuple


class KronIndices(namedtuple('KronIndices',('group_sizes_a', 'group_sizes_b', 'group_indices_a', 'group_indices_b','kron_blocks', 'kron_size','d','elem_groups_a','elem_groups_b'))):
    pass
def kept_blocks_span(bi, num_blocks, d):
    begin = max(0,min(num_blocks,bi-d))
    end = min(num_blocks,bi+d+1)
    return begin,end
class Indices:
    def __repr__(self): return f'<{self.__class__.__name__}(sz:{len(self)}) {self!s}>'
    @property
    def first(self): return self.indices[0] if self else None
    @property
    def last(self): return self.indices[-1] if self else None
    def __bool__(self): return bool(len(self))
    def __iter__(self): return iter(self.indices)
class IndicesContiguous(Indices):
    def __init__(self, begin, end):
        super().__init__()
        self.begin = begin
        self.end = end
    @property
    def indexer(self):
        return slice(self.begin, self.end)
    @property
    def span(self): return self.begin, self.end
    @property
    def indices(self):
        return np.arange(self.begin, self.end)
    def __len__(self):
        return self.end-self.begin
    def __str__(self): return f'[{self.begin}:{self.end}]'
class IndicesStrided2D(Indices):
    def __init__(self, offset,stride,segment_size,num_segments):
        super().__init__()
        self.offset = offset
        self.stride = stride
        self.segment_size = segment_size
        self.num_segments = num_segments
    @property
    def indexer(self):
        num_segs = self.num_segments
        seg_sz = self.segment_size
        ids = np.tile(np.arange(seg_sz),num_segs)+np.repeat(np.arange(num_segs),seg_sz)*self.stride+self.offset
        return ids
    @property
    def end(self): return self.offset + self.stride*self.num_segments + self.segment_size
    indices = indexer
    def __len__(self):
        return self.segment_size*self.num_segments
    def __str__(self): return f'[{self.offset}:{self.num_segments}x[{self.stride}: {self.segment_size}...]:{self.end}]>'
class IndicesAbsolute(Indices):
    def __init__(self, indices):
        self._indices = np.array(indices, int)
    @property
    def indexer(self):
        return self._indices
    indices = indexer
    def __len__(self):
        return len(self._indices)
    def __str__(self):
        k = 5
        tostr = lambda x: ','.join(map(str,x))
        if len(self) <= 2*k:
            sind = tostr(self.indices)
        else:
            sind = f'{tostr(self.indices[:k])},...,{tostr(self.indices[-k:])}'
        return f'[{sind}]'
class KronBlocks(list):
    def __init__(self, num_groups_a,num_groups_b,d, *args):
        self.num_groups_a = num_groups_a
        self.num_groups_b = num_groups_b
        self.d = d
        super().__init__(*args)
    def get_bank_offset(self, i):
        n = self.num_groups_b
        d = self.d
        x = min(i,n+d)
        band = x*n
        top_w = max(0,n-d)
        top_h = max(0,n-d-x)
        top_full = (top_w-1)*top_w/2
        top_tip = (top_h-1)*top_h/2
        top = top_full - top_tip
        bot_w = max(0,x-d)
        bot = (bot_w-1)*bot_w/2
        res = band - bot - top
        return int(res)
    def __getitem__(self, what):
        if isinstance(what, int):
            idx = what
        else:
            i_a,i_b = what
            d = self.d
            off_a = self.get_bank_offset(i_a)
            off_b = i_b - max(0,i_a-d)
            idx = off_a + off_b
        res = super().__getitem__(idx)
        return res
class KronBlock(namedtuple('KronBlock',('group_a','group_b','idx_trunc','idx_group','idx_full'))):
    def __len__(self):
        return len(self.idx_group)
    
def kron_indices(ga, gb, d, issorted=False):
    group_sizes_a = np.bincount(ga)
    group_sizes_b = np.bincount(gb)

    num_groups_a = len(group_sizes_a)
    num_groups_b = len(group_sizes_b)
    
    def get_kept_groups_b(gi_a):
        beg = max(0,gi_a-d)
        end = min(num_groups_b,gi_a+d+1)
        return np.arange(beg,end)
    kron_blocks = KronBlocks(num_groups_a,num_groups_b,d)
    if issorted:
        # ga and gb have elements
        # group is the set of all elements in ga(/gb) that have the same group id
        # Block is the set of all elements in the kroneker product with a given a-group and given b-group.
        # Bank is the union of all B-blocks for a given A-block
        group_cumsize_a = np.pad(np.cumsum(group_sizes_a),[1,0])
        group_cumsize_b = np.pad(np.cumsum(group_sizes_b),[1,0])
    
        def mk_indices(group_sizes):
            ends = np.cumsum(group_sizes)
            begs = np.concatenate([[0],ends[:-1]])
            return [IndicesContiguous(*be) for be in zip(begs,ends)]
        group_indices_a = mk_indices(group_sizes_a)
        group_indices_b = mk_indices(group_sizes_b)
        pos = 0
        offset_trn_a = 0
        for gi_a in range(num_groups_a):
            blk_sz_a = group_sizes_a[gi_a]
            kept_groups_b = get_kept_groups_b(gi_a)
            if not len(kept_groups_b):
                continue
            idxelem_first_kept = group_cumsize_b[kept_groups_b[0]]
            bank_size = group_cumsize_b[kept_groups_b[-1]+1] - idxelem_first_kept
            block_size = bank_size*blk_sz_a
            for i_b,gi_b in enumerate(kept_groups_b):
                blk_sz_b = group_sizes_b[gi_b]
                kron_block_sz = blk_sz_a*blk_sz_b
                block_beg = pos
                pos += kron_block_sz
                block_end = pos
                offset_trn_b = group_cumsize_b[gi_b]-idxelem_first_kept
                idx_trunc = IndicesStrided2D(
                    offset=offset_trn_a + offset_trn_b,
                    stride=bank_size,
                    segment_size=blk_sz_b,
                    num_segments=blk_sz_a)
                idx_group = IndicesContiguous(block_beg, block_end)
                idx_full = IndicesStrided2D(
                    offset=group_cumsize_a[gi_a]*len(gb) + group_cumsize_b[gi_b],
                    stride=len(gb),
                    segment_size=blk_sz_b,
                    num_segments=blk_sz_a)
                kb = KronBlock(group_a=gi_a, group_b=gi_b, idx_trunc=idx_trunc, idx_group=idx_group, idx_full=idx_full)
                kron_blocks.append(kb)
            offset_trn_a += block_size
    else:
        def mk_indices(g, num_groups):
            indices = [[] for _ in range(num_groups)]
            for i, g_i in enumerate(g):
                indices[g_i].append(i)
            return list(map(IndicesAbsolute, indices))
        group_indices_a = mk_indices(ga, num_groups_a)
        group_indices_b = mk_indices(gb, num_groups_b)
        bank_sizes_trn_a = np.empty(num_groups_a,int)
        for gi_a in range(num_groups_a):
            blk_sz_a = group_sizes_a[gi_a]
            if not blk_sz_a:
                bank_sizes_trn_a[gi_a] = 0
            else:
                kept_groups_b = get_kept_groups_b(gi_a)
                bank_size = sum(group_sizes_b[kept_groups_b])
                bank_sizes_trn_a[gi_a] = bank_size
        merge = lambda kept_groups_b: np.fromiter(heapq.merge(*[group_indices_b[gi_b].indices for gi_b in kept_groups_b]),int)
        distribute = lambda data,key,n,base=0: [data[key==i] for i in range(base,base+n)]
        elem_offsets_trn_a = np.pad(np.cumsum(bank_sizes_trn_a[ga]),[1,0])[:-1]
        elem_offsets_trn_groupped_a = distribute(data=elem_offsets_trn_a, key=ga,n=num_groups_a)
        elem_offsets_full_a = np.pad(np.cumsum(np.repeat(len(gb),len(ga))),[1,0])[:-1]
        elem_offsets_full_groupped_a = distribute(data=elem_offsets_full_a, key=ga,n=num_groups_a)
        pos = 0
        for gi_a in range(num_groups_a):
            blk_sz_a = group_sizes_a[gi_a]
            kept_groups_b = get_kept_groups_b(gi_a)
            if not len(kept_groups_b):
                continue
            offset_trn_a = elem_offsets_trn_groupped_a[gi_a]
            offset_full_a = elem_offsets_full_groupped_a[gi_a]
            if not blk_sz_a:
                for gi_b in kept_groups_b:
                    idx_group = IndicesContiguous(pos, pos)
                    idx_trunc = IndicesAbsolute([])
                    idx_full = IndicesAbsolute([])
                    kb = KronBlock(group_a=gi_a, group_b=gi_b, idx_trunc=idx_trunc, idx_group=idx_group, idx_full=idx_full)
                    # print(f'gi_a: {gi_a} gi_b:{gi_b} offset_trn_a: {offset_trn_a} EMPTY')
                    kron_blocks.append(kb)
            else:
                elems_idx_merged_b = merge(kept_groups_b)
                elems_grp_merged_b = gb[elems_idx_merged_b]
                elems_idxfull_b = distribute(elems_idx_merged_b,key=elems_grp_merged_b,n=len(kept_groups_b),base=kept_groups_b[0])
                num_elems_b = len(elems_idx_merged_b)
                elems_idxtrunc_b = distribute(np.arange(num_elems_b),key=elems_grp_merged_b,n=len(kept_groups_b),base=kept_groups_b[0])
                for i_b,gi_b in enumerate(kept_groups_b):
                    blk_sz_b = group_sizes_b[gi_b]
                    kron_block_sz = blk_sz_a*blk_sz_b
                    block_beg = pos
                    pos += kron_block_sz
                    block_end = pos
                    idx_group = IndicesContiguous(block_beg, block_end)
                    
                    off_trunc_b = elems_idxtrunc_b[i_b]
                    idx_trunc = IndicesAbsolute(np.repeat(offset_trn_a, blk_sz_b) + np.tile(off_trunc_b,blk_sz_a))
                    
                    off_full_b = elems_idxfull_b[i_b]
                    idx_full = IndicesAbsolute(np.repeat(offset_full_a, blk_sz_b) + np.tile(off_full_b,blk_sz_a))
                    # print(f'gi_a: {gi_a} gi_b:{gi_b} offset_trn_a: {offset_trn_a} offset_full_a: {offset_full_a} off_trunc_b:{off_trunc_b} off_full_b:{off_full_b}')
                    
                    kb = KronBlock(group_a=gi_a, group_b=gi_b, idx_trunc=idx_trunc, idx_group=idx_group, idx_full=idx_full)
                    kron_blocks.append(kb)
    kron_size = pos
    return KronIndices(
        group_sizes_a=group_sizes_a, group_sizes_b=group_sizes_b, group_indices_a=group_indices_a, group_indices_b=group_indices_b, kron_blocks=kron_blocks, kron_size=kron_size,d=d,
        elem_groups_a=ga, elem_groups_b=gb)

def tkron(A,ga, B, gb, d=0, group=False, issorted=False):
    ki_i = kron_indices(ga,gb,d=d,issorted=issorted)
    nk_i = ki_i.kron_size
    ki_j = ki_i
    nk_j = ki_j.kron_size
    K = np.zeros((nk_i,nk_j))
    indices = 'idx_group' if group else 'idx_trunc'
    sm = lambda M,ids_i,ids_j: M[ids_i,ids_j] if isinstance(ids_i, slice) or isinstance(ids_j,slice) else M[np.ix_(ids_i,ids_j)]
    for kb_i in ki_i.kron_blocks:
        ids_ki = getattr(kb_i, indices).indexer 
        ids_ai = ki_i.group_indices_a[kb_i.group_a].indexer
        ids_bi = ki_i.group_indices_b[kb_i.group_b].indexer
        if len(kb_i):
            for kb_j in ki_j.kron_blocks:
                ids_kj = getattr(kb_j,indices).indexer
                ids_aj = ki_j.group_indices_a[kb_j.group_a].indexer
                ids_bj = ki_j.group_indices_b[kb_j.group_b].indexer
                if len(kb_j):
                    Ablk = sm(A,ids_ai,ids_aj)
                    Bblk = sm(B,ids_bi,ids_bj)
                    # print(f'For block: A{kb_i.group_a},{kb_j.group_a} B{kb_i.group_b},{kb_j.group_b}:')
                    # print(ids_ai, ids_aj, ids_bi, ids_bj, ids_ki, ids_kj)
                    Ksrc = np.kron(Ablk, Bblk)
                    if isinstance(ids_ki, slice) or isinstance(ids_kj,slice):
                        K[ids_ki,ids_kj] = Ksrc
                    else:
                        K[np.ix_(ids_ki,ids_kj)] = Ksrc    
    return K



class IndexedValue(NamedTuple):
    index: int
    value: int
    def __repr__(self):
        return f'IV({self.index}: {self.value})'
'''
def scaled_block_row_matrix_times_vector(A,x,block_cumsums:List[IndexedValue],w,out=None):
    ''
    @param w: a weight vector indexed by block indices
    ''
    m,n = A.shape
    b = out if out is not None else np.empty(m, dtype=x.dtype)
    b[:] = 0

    num_block_cols = len(block_cumsums)-1
    elem_inn_off = block_cumsums[0].value
    blk_inn_off = block_cumsums[0].index
    
    elem_inn_beg = 0
    for binz_inn in range(num_block_cols):
        bi_inn = block_cumsums[binz_inn].index
        elem_inn_end = block_cumsums[binz_inn+1].value - elem_inn_off
        bri_inn = bi_inn - blk_inn_off
        b += x[elem_inn_beg:elem_inn_end]@A[elem_inn_beg:elem_inn_end,:]*w[bri_inn]
        elem_inn_beg = elem_inn_end
    assert elem_inn_end == m,f'Block sizes sum to {elem_inn_end} instead of vector size: {m}.'
    return b
'''

def scaled_block_row_matrix_times_vector(A,x,block_cumsums:List[IndexedValue],wnz,out=None):
    '''
    @param w: a weight vector indexed by block indices
    '''
    m,n = A.shape
    b = out if out is not None else np.empty(m, dtype=x.dtype)
    b[:] = 0

    num_block_cols = len(block_cumsums)-1
    elem_inn_off = block_cumsums[0].value
    # blk_inn_off = block_cumsums[0].index
    
    elem_inn_beg = 0
    elem_inn_end = 0 # TODO: Remove (for assert)
    for binz_inn in range(num_block_cols):
        # bi_inn = block_cumsums[binz_inn].index
        elem_inn_end = block_cumsums[binz_inn+1].value - elem_inn_off
        # bri_inn = bi_inn - blk_inn_off
        b += x[elem_inn_beg:elem_inn_end]@A[elem_inn_beg:elem_inn_end,:]*wnz[binz_inn]
        elem_inn_beg = elem_inn_end
    assert elem_inn_end == m,f'Block sizes sum to {elem_inn_end} instead of vector size: {m}.'
    return b

    
class BankBlocks(NamedTuple):
    idx_elem_inn: IndicesContiguous
    idx_elem_out: IndicesContiguous
    idx_block_inn: IndicesContiguous
    idx_blknz_inn: IndicesContiguous
    block_out: int
    block_cumsizes: List[IndexedValue]
    pbbm:'PackedBlockBandedMatrix'
    bnzi:'PackedBlockBandedMatrix._BankNZIndices'
    class BankVector(NamedTuple):
        blocks: 'BankBlocks'
        idx_relative:int
        idx_element:int
        def __repr__(self):
            return f'<{self.blocks.__class__.__name__}Bank({self.idx_relative}->{self.idx_element}) sz:{len(self.idx_data)}>'
        @property
        def idx_data(self):
            return self.blocks.idx_data(self.idx_relative)
        @property
        def data(self):
            return self.blocks.pbbm.data[self.idx_data.indexer]
    @property
    def elems(self):
        for ri,ei in enumerate(self.idx_elem_out.indices):
            yield self.BankVector(self,ri,ei)
    @property
    def num_elems_inn(self): return self.block_cumsizes[-1].value - self.block_cumsizes[0].value
    @property
    def num_elems_out(self): return len(self.idx_elem_out)
    @property
    def data(self):
        return self.pbbm.data[self.idx_data().indexer].reshape(self.num_elems_out, self.num_elems_inn)
    def __repr__(self):
        span = lambda x:f'{x.begin}:{x.end}'
        return f'<{self.__class__.__name__}({self.block_out},{self.idx_blknz_inn.indices}) w/ {self.num_elem_rows}x{self.num_elem_columns} elems: [{span(self.idx_elem_rows)}x{span(self.idx_elem_columns)}]>'
class BlockColumn(BankBlocks):
    idx_elem_rows = BankBlocks.idx_elem_inn
    idx_elem_columns = BankBlocks.idx_elem_out
    num_elem_rows = BankBlocks.num_elems_inn
    num_elem_columns = BankBlocks.num_elems_out
    idx_block_rows = BankBlocks.idx_block_inn
    block_row = BankBlocks.block_out
    @property
    def data(self):
        return super().data.T
    def idx_data(self, ri = None):
        bnzi = self.bnzi
        col_size = len(self.idx_elem_rows)
        bank_offset = self.pbbm.column_bank_elem_offsets[bnzi.out]
        if ri is None:
            begin, end = bank_offset, bank_offset+col_size*self.num_elem_columns
        else:
            begin = bank_offset+col_size*ri
            end = begin + col_size
        return IndicesContiguous(begin, end)
class BlockRow(BankBlocks):
    idx_elem_rows = BankBlocks.idx_elem_out
    idx_elem_columns = BankBlocks.idx_elem_inn
    num_elem_rows = BankBlocks.num_elems_out
    num_elem_columns = BankBlocks.num_elems_inn
    idx_block_cols = BankBlocks.idx_block_inn
    block_row = BankBlocks.block_out
    def __init__(self,*args,**kwargs):
        super().__init__()
        # Proceed in an echelon fashion the nz row blocks along each column.
        bnzi = self.bnzi
        pbbm = self.pbbm
        d = pbbm.d
        row_cszs = pbbm.row_cumsizes
        col_cszs = pbbm.column_cumsizes
        num_blk_rows = pbbm.num_block_rows
        
        
        offset_left = self.pbbm.column_bank_elem_offsets[bnzi.inn_beg]
        
        # our offsets
        offset_row_beg = row_cszs[bnzi.out].value
        offset_row_end = row_cszs[bnzi.out+1].value
        blk_m = offset_row_end - offset_row_beg
        blk_n = len(self.idx_elem_columns)
        
        indices0 = np.zeros(blk_n)

        map_bnzi2bi = lambda x: col_cszs[x].index
        bri_col = 0
        bi_row = self.block_row
        bnzi_col = bnzi.inn_beg
        bi_col = map_bnzi2bi(bnzi_col)
        last_col_of_cur_blk = -1

        offset_current = offset_left
        for eri_col,ei_col in enumerate(self.idx_elem_columns):
            if ei_col >= last_col_of_cur_blk:
                last_col_of_cur_blk = self.block_cumsizes[bri_col+1].value
                # bi_row_kept_beg = max(0,bi_col-d)
                # bi_row_kept_end = min(num_blk_rows,bi_col+d+1) 
                bi_row_kept_beg, bi_row_kept_end = kept_blocks_span(bi=bi_col, num_blocks=num_blk_rows,d=d) 
                
                if bi_row_kept_beg>num_blk_rows:
                    break
                bnzi_row_kept_beg = pbbm.map_row_bi2bnzi[bi_row_kept_beg]
                bnzi_row_kept_end = pbbm.map_row_bi2bnzi[bi_row_kept_end]
                offset_above = offset_row_beg - row_cszs[bnzi_row_kept_beg].value
                offset_below = row_cszs[bnzi_row_kept_end].value - offset_row_end
                bri_col += 1
                bnzi_col += 1
                bi_col = map_bnzi2bi(bnzi_col)
            offset_current += offset_above
            indices0[eri_col] = offset_current
            offset_current += blk_m + offset_below
        
        self.indices0 = indices0
    
    def idx_data(self, ri=None):
        if ri is None:
            neo,nei = self.num_elems_out, self.num_elems_inn
            idx = np.tile(self.indices0, neo) + np.repeat(np.arange(neo), nei)
        else:
            idx = self.indices0 + ri
        return IndicesAbsolute(idx.astype(int))

_BankNZIndices = namedtuple('_BankNZIndices',('inn_beg', 'inn_end', 'out'))
    
class PackedBlockBandedMatrix(NamedTuple):
    m: int
    n: int
    d: int
    column_sizes: List[IndexedValue]
    row_sizes: List[IndexedValue]
    column_cumsizes:List[IndexedValue]
    row_cumsizes:List[IndexedValue]
    column_cumsizes:List[IndexedValue]
    column_bank_elem_offsets:np.ndarray # the location in data of first element in bank-column (nz indexed)
    map_row_bi2bnzi:np.ndarray 
    data:np.ndarray
    w:np.ndarray
    @property
    def dtype(self): return self.data.dtype
    @property
    def num_block_rows(self): return self.row_cumsizes[-1].index
    @property
    def num_block_columns(self): return self.column_cumsizes[-1].index
    '''
    @property
    def effective_m(self): return self.row_cumsizes[-1].value
    @property
    def effective_n(self): return self.column_cumsizes[-1].value
    '''
    @property
    def kron_size(self): return self.column_bank_elem_offsets[-1]
    @classmethod
    def from_data(cls, ga, gb, d, data, w):
        def cumsum(index) -> List[IndexedValue]:
            indices, values = zip(*index)
            res = np.cumsum(values)
            return list(map(IndexedValue, zip(indices, res)))
        def compress(elem_groups):
            counts = np.bincount(elem_groups)
            szs = []
            cszs = []
            csum = 0
            for gi, gsz in enumerate(counts):
                if gsz == 0: continue
                szs.append(IndexedValue(gi,gsz))
                cszs.append(IndexedValue(gi,csum))
                csum += gsz
            cszs.append(IndexedValue(gi+1,csum))
            return szs, cszs
        
        col_szs, col_cszs = compress(ga)
        row_szs, row_cszs = compress(gb)
        
        '''maps to each (absolute) index the count of nz blocks before it.'''
        map_row_bi2bnzi = np.zeros(row_cszs[-1].index+1,int)
        map_row_bi2bnzi[[r.index+1 for r in row_cszs[:-1]]] = 1
        map_row_bi2bnzi = np.cumsum(map_row_bi2bnzi)
        
        column_bank_elem_offsets = [0]
        offset = 0
        for bnzi in cls._iter_banks(row_cszs, col_cszs, d):
            blk_m = row_cszs[bnzi.inn_end].value - row_cszs[bnzi.inn_beg].value
            blk_n = col_cszs[bnzi.out+1].value - col_cszs[bnzi.out].value
            # bank_elem_offset = _BankElemOffset(bnzi_row_beg, bnzi_row_end, offset)
            offset += blk_m*blk_n
            column_bank_elem_offsets.append(offset)
            print(f'{bnzi.out}: {bnzi.inn_beg}({row_cszs[bnzi.inn_beg].index}):{bnzi.inn_end}({row_cszs[bnzi.inn_end].index})')
            print(f'{bnzi.out}: {bnzi.inn_beg}({row_cszs[bnzi.inn_beg].value}):{bnzi.inn_end}({row_cszs[bnzi.inn_end].value})')
        # column_bank_elem_offsets.append(_BankElemOffset(offset,None,None))
        column_bank_elem_offsets = np.array(column_bank_elem_offsets)
        m = len(gb)
        n = len(ga)
        return cls(m=m,n=n,d=d,data=data,w=w,
                   column_sizes=col_szs, column_cumsizes=col_cszs,
                   row_sizes=row_szs, row_cumsizes=row_cszs,
                   column_bank_elem_offsets=column_bank_elem_offsets,
                   map_row_bi2bnzi=map_row_bi2bnzi)
    
    @classmethod
    def _iter_banks(cls,inn_csizes, out_csizes, d):
        num_inn_blocks = inn_csizes[-1].index # Always exists
        
        bnzi_inn_beg = 0
        bi_inn_beg = inn_csizes[bnzi_inn_beg].index
        bnzi_inn_end = 0
        bi_inn_end = inn_csizes[bnzi_inn_end].index
        bnzi_out = 0
        for bi_o, _ in out_csizes[:-1]:
            # This is for upkeeping the current banks with jumps
            # first_kept_inn = max(0,bi_o-d) # TODO: DBGREMOVE
            first_kept_inn,end_kept_inn = kept_blocks_span(bi=bi_o, num_blocks=num_inn_blocks, d=d)
            while bi_inn_beg < first_kept_inn:
                if bi_inn_beg >= num_inn_blocks:
                    return
                bnzi_inn_beg += 1 # Iterator advance
                bi_inn_beg = inn_csizes[bnzi_inn_beg].index
            #end_kept_inn = min(bi_o+d+1,num_inn_blocks) # TODO: DBGREMOVE
            while bi_inn_end < end_kept_inn:
                bnzi_inn_end += 1 # iterator advance
                bi_inn_end = inn_csizes[bnzi_inn_end].index
            print(f'EKI: {first_kept_inn} - {end_kept_inn}')
            yield _BankNZIndices(out=bnzi_out,inn_beg=bnzi_inn_beg,inn_end=bnzi_inn_end)
            bnzi_out += 1
    def block_banks(self,inner_rows):
        d = self.d
        if inner_rows: # Iterate first over rows (vertically): Banks are rows
            inn_cszs = self.row_cumsizes
            out_cszs = self.column_cumsizes
            num_blocks_inn = self.num_block_rows
        else:
            inn_cszs = self.column_cumsizes
            out_cszs = self.row_cumsizes
            num_blocks_inn = self.num_block_columns
        
        for bnzi in self._iter_banks(inn_cszs, out_cszs, d):
            idx_elem_inn = IndicesContiguous(inn_cszs[bnzi.inn_beg].value,inn_cszs[bnzi.inn_end].value)
            idx_elem_out = IndicesContiguous(out_cszs[bnzi.out].value,out_cszs[bnzi.out+1].value)
            bi_out = out_cszs[bnzi.out].index
            idx_block_inn = IndicesContiguous(*kept_blocks_span(bi=bi_out, num_blocks=num_blocks_inn, d=d))
            idx_blknz_inn = IndicesAbsolute([iv.index for iv in inn_cszs[bnzi.inn_beg:bnzi.inn_end]])
            block_out = bi_out
            block_cumsizes = inn_cszs[bnzi.inn_beg:bnzi.inn_end+1]
            cls = self.__class__
            cls_sub = BlockColumn if inner_rows else BlockRow
            yield cls_sub(
                idx_elem_inn=idx_elem_inn, idx_elem_out=idx_elem_out,
                idx_block_inn=idx_block_inn, idx_blknz_inn=idx_blknz_inn,
                block_out=block_out, block_cumsizes=block_cumsizes,pbbm=self, bnzi=bnzi
                )
             
    @property
    def block_rows(self): return self.block_banks(False)
    @property
    def block_columns(self): return self.block_banks(True)
    
def row_vector_times_packed_block_toeplitz_hadamard_matrix(x, A : PackedBlockBandedMatrix, w, out=None):
    '''
    @param w: a 2*d+1 sized weight vector, indexed by block distances.
    '''
    em,en = A.m, A.n
    bm,bn = len(A.column_sizes), len(A.row_sizes)
    d = A.d
    assert len(w)==2*d+1, f'The size of the weights vector is {len(w)} instead of {2*d+1}'
    b = out if out is not None else np.zeros(en, dtype=x.dtype)
    if not A.column_sizes:
        return
    for block_inn in A.block_columns:
        bank_A = block_inn.data
        bank_x = x[block_inn.idx_elem_rows.indexer]
        bank_col_cszs = block_inn.block_cumsizes
        bi_col_beg, bi_col_end = block_inn.idx_block_rows.span
        bi_row = block_inn.block_row
        # bank_w = w[bi_col_beg+d-bi_row:bi_col_end+d-bi_row]
        bank_w = w[block_inn.idx_blknz_inn.indices+d-bi_row]
        bank_b = np.empty(block_inn.num_elems_out)
        scaled_block_row_matrix_times_vector(A=bank_A,x=bank_x,block_cumsums=bank_col_cszs, wnz=bank_w,out=bank_b)
        b[block_inn.idx_elem_out.indexer] = bank_b
    return b

def row_vector_of_block_toeplitz_times_matrix_times_vector(A,x,w_nz,block_cumsizes:List[IndexedValue],out=None):
    m,n = A.shape
    assert len(x)==n,f'Size mismatch: vector x has length {len(x)} instead of {n}.'
    '''l is the output size'''
    l = block_cumsizes[-1].value - block_cumsizes[0].value
    if out is None:
        b = np.empty(m,dtype=x.dtype)
    else:
        b = out
        assert b.ndim == 1,f'b has {b.ndim} dimensions instead of 1'
        assert len(b)==l,f'b has size {len(b)} instead of {l}'
    num_blk = len(block_cumsizes)-1
    bi_off = block_cumsizes[0].index
    ei_off = block_cumsizes[0].value
    ei_beg = 0
    for bnzi in range(num_blk):
        bi = block_cumsizes[bnzi].index
        bri = bi - bi_off # Relative offset within current sub-matrix
        ei_end = block_cumsizes[bnzi+1].value - ei_off # Remove offset from cumsize
        b_row = (A[ei_beg:ei_end,:]@x)*w_nz[bnzi]
        b[ei_beg:ei_end] = b_row
        ei_beg = ei_end
    return b



def tkronvtrn(A,B,X:PackedBlockBandedMatrix, out=None):
    Bx = np.empty(B.shape[1],dtype=X.dtype)
    d = X.d
    w = X.w
    '''
    data: np.ndarray
    m: int
    n: int
    bandwidth: int
    column_sizes: List[IndexedValue]
    row_sizes: List[IndexedValue]
    '''
    if out is None:
        y = np.empty(X.kron_size,dtype=X.dtype)
    else:
        y = out
        assert len(y)==X.kron_size,f'Size of output is {len(y)} instead of {X.kron_size}'
    for bank_row in X.block_rows:
        print(bank_row)
        bi_row = bank_row.block_row
        for elem_row in bank_row.elems:
            print(f'  {elem_row}')
            b = B[elem_row.idx_element,:] # don't need effective size here.
            Bx = row_vector_times_packed_block_toeplitz_hadamard_matrix(A=X, w=w, x=b)
            # @TODO: need effective size here
            blk_col_beg,blk_col_end = bank_row.idx_block_inn.span
            elem_col_beg,elem_col_end = bank_row.idx_elem_inn.span
            y_row = np.zeros(len(elem_row.idx_data),dtype=y.dtype)
            A_row = A[elem_col_beg:elem_col_end,:]
            x_row = Bx
            blk_cszs_row = bank_row.block_cumsizes
            wnz_row = w[bank_row.idx_blknz_inn.indices+d-bi_row]
            row_vector_of_block_toeplitz_times_matrix_times_vector(A=A_row,x=x_row,block_cumsizes=blk_cszs_row,w_nz=wnz_row,out=y_row)
            y[elem_row.idx_data.indexer] = y_row
    return y

def tkronvfull(A,B,x,ki:KronIndices,w):
    x_trn = np.empty(ki.kron_size,x.dtype)
    kb: KronBlock
    for kb in ki.kron_blocks:
        ids_src = kb.idx_full.indexer
        ids_dst = kb.idx_trunc.indexer
        x_trn[ids_dst] = x[ids_src]
    X = PackedBlockBandedMatrix.from_data(ga=ki.elem_groups_a, gb=ki.elem_groups_b, d=ki.d,data=x_trn, w=w)
    return tkronvtrn(A=A, B=B, X=X)

