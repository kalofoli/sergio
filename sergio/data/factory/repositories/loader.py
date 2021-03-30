'''
Created on Mar 30, 2021

@author: janis
'''

import pandas
from sergio.data.factory import read_url, read_csv, save_url
from colito.logging import getModuleLogger
import os
import zipfile

log = getModuleLogger(__name__)

class RepositoryLoader():
    __name__ = None
    
    
    def __init__(self, data_home='.'):
        self.data_home = data_home
        self._metadata = None
    
    @property
    def meta_file(self): return os.path.join(self.data_home,f'{self.__name__}-datasets.csv')
    @property
    def metadata(self): return self.load_metadata(force_reload=False)
    
    def parse_dataset_data(self, name, path):
        raise NotImplementedError('Override')
    
    def load_dataset(self, name):
        url = self.get_dataset_url(name)
        with self.fetch_zip_cached(url, name) as zip_ref:
            log.info(f'Parsing dataset {name}')
            dataset = self.parse_dataset_data(name, path=zip_ref)
        
        #dataset.data = SliceableList(dataset.data)
        return dataset

    def list_datasets(self):
        if self.metadata is None:
            self.load_metadata()
        return list(self.metadata.index)

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
            df = self._download_metadata()
            if cache and meta_file is not None:
                df.to_csv(meta_file, sep='\t')
            return df
        if force_reload:
            df = download()
        else:
            if self._metadata is not None:
                df = self._metadata
            else:
                log.info(f'Trying to load cache: {meta_file}')
                try:
                    df = pandas.read_csv(meta_file, sep='\t').set_index('name')
                except Exception as e:
                    log.info(f'Could not load cache: {meta_file} because of: {e}')
                    df = download()

        self.__class__.metadata = df
        return self.metadata
