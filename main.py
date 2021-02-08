'''
Created on Jan 31, 2021

@author: janis
'''


from colito.logging import getModuleLogger, setAdapterFactory, SergioLogger

setAdapterFactory(SergioLogger)
log = getModuleLogger(__name__)


if __name__ == '__main__':
    from sergio.sd.datasets.morris import MorrisLoader
    log.add_stderr()
    log.setLevel('INFO')
    
    ml = MorrisLoader()
    ds = ml.fetch_dataset('MUTAG') 
    print('Hello World')