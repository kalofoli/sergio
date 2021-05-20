

from collections import namedtuple

import numpy as np
import datetime
import json
import requests
import inspect
import sqlite3

from tqdm import tqdm
import pandas as pd

from colito.logging import getModuleLogger
log = getModuleLogger(__name__)

def command(command, v=3, parts = [], prep=None, args=[], kwargs={}):
    base_url = f'https://financialmodelingprep.com/api/v{v}/{command}'
    def decorator(fn):
        def mk_par(arg, val=inspect.Parameter.empty):
            return inspect.Parameter(arg, kind=inspect._ParameterKind.POSITIONAL_OR_KEYWORD, default=val)
        pars = [mk_par('self'), *[mk_par(a) for a in parts], *[mk_par(a) for a in args], *[mk_par(a,v) for a,v in kwargs.items()]]
        sig = inspect.Signature(pars)
        def wrapper(self, *args, **kwargs):
            pars = sig.bind(self, *args, **kwargs)
            pars.apply_defaults()
            params = dict(pars.arguments)
            part_vals = [params.pop(p) for p in parts]
            url = '/'.join([base_url, *part_vals])
            params.pop('self')
            text = self.fetch(url, params=params)
            data = json.loads(text)
            if prep is not None:
                data = prep(data)
            return fn(self, data)
        wrapper.__signature__ = sig
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator

AttrInfo = namedtuple('AttrInfo', ('name', 'kind', 'selected','conv', 'pp')) # pp is what must happen before using in sergio
class FinancialModellingPrep:
    
    def __init__(self, apikey, db):
        self._apikey = apikey
        self._db = db
        self._session = requests.Session()
    def fetch_single(self, sql, params=[]):
        rows = self._db.execute(sql, params).fetchall()
        return [row[0] for row in rows]
    def __del__(self):
        self._db.close()

    @classmethod
    def profile_attr_info(cls):
        attr_infos = []

        def addinf(names, kind, conv=None, selected=True, pp=None):
            nonlocal attr_infos
            attr_infos += [AttrInfo(name=name, kind=kind, selected=selected, conv=conv, pp=pp) for name in names]

        addinf(['price', 'beta',  'lastDiv', 'changes', 'dcfDiff', 'dcf'], 'NUMERIC', float)
        addinf(['volAvg', 'mktCap', 'fullTimeEmployees'], 'NUMERIC', int)
        addinf(['ipoDate'], 'NUMERIC', cls.parse_date, pp=datetime.date.toordinal)
        addinf(['cik', 'cusip'], 'INDEX', str)
        addinf(['sector', 'country'], 'CATEGORICAL')
        addinf(['isEtf', 'isActivelyTrading'], 'BOOLEAN', bool)
        addinf(['zip', 'website', 'description', 'ceo', 'phone', 'address', 'image', 'defaultImage', 'range'], None)
        return attr_infos

    __attr_infos__ = None
    @classmethod
    def _apply_attr_info(cls, which, row):
        attr_infos = cls.__attr_infos__[which]
        for ai in attr_infos:
            if ai.conv is not None:
                val = row[ai.name]
                if val is not None:
                    try:
                        row[ai.name] = ai.conv(val)
                    except Exception as e:
                        log.error(f'While converting {ai} with value {val}')
        
    @classmethod
    def parse_date(what):
        if isinstance(what, datetime.date):
            return what
        elif isinstance(what, str):
            return datetime.date.fromisoformat(what)
        else:
            raise TypeError(f'Could not parse date from {what} of type {type(what).__name__}.')

    @property
    def stocks(self): return pd.read_sql_query('select * from stocks', self._db)
    @property
    def company_profiles(self): return pd.read_sql_query('select * from company_profile', self._db)
    @property
    def tables(self):
        return self.fetch_single('select name from SQLITE_MASTER where type="table"')
    
    EXCHANGES = 'ETF|MUTUAL_FUND|COMMODITY|INDEX|CRYPTO|FOREX|TSX|AMEX|NASDAQ|NYSE|EURONEXT|XETRA|NSE|LSE'.split('|')
    def fetch(self, url, params):
        params = {'apikey':self._apikey, **params}
        with self._session.get(url, params=params) as resp:
            return resp.text
    @command('dowjones_constituent', v=3, prep=pd.DataFrame)
    def dowjones_constituent(self, result):
        return result
    @command('actives', v=3, prep=pd.DataFrame)
    def most_active(self, result):
        return result
    @command('search-ticker', v=3, prep=pd.DataFrame, args=['query','exchange'])
    def search_ticker(self, result):
        return result
    @command('profile', parts=['symbol'])
    def company_profile(self, result):
        if result:
            row = result[0]
            self._apply_attr_info('company_profile', row)
            return row

    @command('historical-price-full', parts=['symbol'])
    def historical_price_full(self, result):
        if result:
            symbol = result['symbol']
            df = pd.DataFrame(result['historical'])
            df = df.astype({'date':'datetime64'}).drop('label',1).assign(symbol=symbol)
            return df
        else:
            return None
    @command('historical-price-full', parts=['symbol'])
    def historical_price_full_raw(self, result):
        if result:
            symbol = result['symbol']
            return [{'symbol':symbol,**row} for row in result['historical']]
        else:
            return None
    @command('key-executives', parts=['symbol'])
    def key_executives(self, result):
        if result:
            df = pd.DataFrame(result)
            return df
        else:
            return None
    def store_into(self, args, key, table, loader, batch_size=10, single=True):
        '''Store a loaded value into a db
        
        :param single: The result is a single entry or a df
        '''
        from itertools import chain
        tables = self.tables
        if table in tables:
            keys_stored = self.fetch_single(f'select distinct `{key}` from `{table}`;')
            args_left = [a for a in args if a not in keys_stored]
        else:
            args_left = args
        it = iter(enumerate(tqdm(args_left)))
        def get_batch():
            results = []
            while len(results) < batch_size:
                try:
                    idx, arg = next(it)
                except StopIteration:
                    break
                try:
                    result = loader(arg)
                    if result is None or single and not result:
                        log.error(f'EMPTY: While loading {idx} with arguments: {arg}')
                    else:
                        results.append(result)
                except Exception as e:
                    log.error(f'ERROR: While loading {idx} with arguments: {arg}: {e}', exc_info=True)
            if not results:
                return None
            else:
                return results if single else list(chain(*results))
        if table not in tables:
            rows = get_batch()
            df = pd.DataFrame(rows)
            df.to_sql(table, self._db, index=False)

        def db_insert(rows):
            sparam = ','.join(['?']*len(rows[0]))
            sql = f'insert into `{table}` values ({sparam})'
            self._db.executemany(sql, [list(row.values()) for row in rows])
            self._db.commit()
        while True:
            df = get_batch()
            if df is None:
                break
            db_insert(df)

FinancialModellingPrep.__attr_infos__ = {'company_profile': FinancialModellingPrep.profile_attr_info()}
            

class StocksLoader:
    def __init__(self, db, price_since = '2020-01-01', price_until = '2021-01-01', today = '2021-01-01', ipo_until=None, price_from='vwap', title_until=None, same_times=True):
        '''
        :param_same_times: used in structures. Keeps only price data over the most frequent date sequence.'''
        self._db = db
        self._today = today
        self._price_since = price_since
        self._price_until = price_until
        self._ipo_until = ipo_until if ipo_until is not None else today
        self._title_until = title_until if title_until is not None else today
        self._price_from = price_from
        self._same_times = same_times
    @property
    def tables(self):
        rows = self._db.execute('select name from SQLITE_MASTER where type="table";').fetchall()
        return [row[0] for row in rows]
    @classmethod
    def days_since(cls, x, date):
        s = pd.Series(x).astype(np.datetime64)
        return (s - np.datetime64(date)).dt.days

    def get_profiles(self):
        df_profs = pd.read_sql_query('select * from company_profile order by symbol', self._db)\
            .assign(
                daysSinceIpo = lambda x: -self.days_since(x.ipoDate, self._ipo_until)
            )
        return df_profs
    def get_part_profiles(self):
        #import warnings
        #warnings.simplefilter("error")
        df = self.get_profiles().drop(['ipoDate'],1).drop([
            'description','ceo','phone','address','city','state','zip','companyName',
            'image','defaultImage','cik','cusip','isin','exchange',
            'range','changes','exchangeShortName','website',
            'currency' # non-descriptive
            ],axis=1)
        log_feats = ['mktCap','volAvg']
        for feat in log_feats:
            val = df[feat].copy()
            val[val==0] = np.nan
            df[f'{feat}Log10'] = np.log10(val)
        df['fullTimeEmployees'] = df.fullTimeEmployees.replace('',np.nan)
        df = df.astype({'isEtf':bool,'isActivelyTrading':bool})
        df = df.drop(log_feats,1)
        df.assign()
        df = df.set_index('symbol', drop=True)
        return df
    def get_prices(self, since, until, value='vwap'):
        swhich = f'symbol,date,`{value}`' if value is not None else '*'
        df = pd.read_sql_query(
                f'select {swhich} from prices_daily where date>:since and date<:until',
                self._db,
                params={'since':since,'until':until}
            )\
            .assign(days = lambda x: self.days_since(x.date, since)).drop('date',1)
        return df
    def get_part_structures(self, same_times=None):
        if same_times is None:
            same_times = self._same_times
        price_from = self._price_from
        df = self.get_prices(since=self._price_since, until=self._price_until, value=price_from)
        def widen(df):
            x = df.days.values
            y = df[price_from].values
            p = np.argsort(x)
            data = x[p], y[p]
            return pd.DataFrame([{'x':x[p], 'y':y[p], 'n':len(p)}])
        df = df.groupby('symbol').apply(widen)
        if same_times:
            len_freq = np.bincount(df.n)
            sz = np.argmax(len_freq)
            df = df[df.n == sz]
            assert len(set(tuple(x) for x in df.x))==1, f'Non equal x. Need filtering.'
            log.info(f'Keeping only most frequent size: {sz} (freq: {len_freq[sz]} or {len_freq[sz]/len_freq.sum()*100:.2f}%)')
            df = df.drop(['x','n'],1).droplevel(1).rename({'y':'prices'},axis=1)
        return df
    def get_executives(self, title_until):
        df = pd.read_sql_query('select * from key_executives order by symbol', self._db)
        df = df.assign(seniority=lambda x:-self.days_since(x.titleSince, title_until))
        return df
    def get_part_executives(self):
        df = self.get_executives(title_until=self._title_until)
        def ongroup(x):
            num_doctors = x['name'].str.match('Dr[.]').sum()
            idl_age = ~x.yearBorn.isna()
            avg_age = 2021-x.yearBorn[idl_age].mean() if idl_age.any() else None
            data = {
                'female':(x.gender=='female').sum(), 'num_doctors': num_doctors,
                'avg_age':avg_age, 'avg_seniority': np.nanmean(x.seniority),
                'num_listed':x.shape[1],'max_pay':x.pay.max()}
            return pd.DataFrame([data])
        df = df.groupby('symbol')
        df = df.apply(ongroup).droplevel(1)
        assert np.all(df.num_listed==9),f'Unequal number of lsited executives for a company'
        df = df.drop(['num_listed'],1)
        return df
    _parts = ['executives', 'profiles', 'structures']

    def assemble(self, parts=None, merge='inner'):
        if parts is None:
            parts = self._parts
        def get_part(part):
            log.info(f'Loading part {part}')
            df = getattr(self, f'get_part_{part}')()
            log.info(f'Found {df.shape[0]} rows and {df.shape[1]} columns.')
            return df
        df = get_part(parts[0])
        for part in parts[1:]:
            df_part = get_part(part)
            df = pd.merge(df, df_part, on='symbol', how=merge)
            log.info(f'After merge: {df.shape[0]} rows remain.')
        return df

    def make_data(self, parts=None, merge='inner', same_time=True):
        if parts is None:
            parts = self._parts
        df = self.assemble(parts=parts, merge=merge)

        df_structs = pd.DataFrame(df.pop('prices'))
        sparts = ''.join(f'{part[0].upper()}{part[1:].lower()}' for part in parts)
        stime = 'SameTime' if same_time else 'AllTimes'
        tag = f'{merge.lower()}-{sparts}-{stime}'

        from sergio.data.bundles import EntityAttributesWithStructures
        eas = EntityAttributesWithStructures(attribute_data=df, name=f'stock-{tag}', structures=df_structs)
        return eas    
