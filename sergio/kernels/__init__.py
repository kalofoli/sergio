'''
Measures

Created on Oct 3, 2019

@author: janis
'''
from colito.summaries import Summarisable, DEFAULT_SUMMARY_OPTIONS,\
    SummaryOptions, SummarisableDict, SummarisableAsDict
from colito.collection import ClassCollection, ClassCollectionFactoryRegistrar


KERNELS = ClassCollection('Kernels')

class Kernel(SummarisableAsDict, ClassCollectionFactoryRegistrar):
    __collection_factory__ = KERNELS
    __collection_tag__ = None
    
    __kernel_params__ = ()

    def fit(self, X):
        self._X = X
        return self
    
    def transform(self, Y):
        raise NotImplementedError('Override to use')
    
    def fit_transform(self, X, Y=None):
        if Y is None:
            Y = X
        return self.fit(X).transform(Y)

    def __summary_dict__(self, options:SummaryOptions=DEFAULT_SUMMARY_OPTIONS):
        return self.get_params()

    def get_params(self, keys=None):
        if keys is None:
            keys = self.__kernel_params__
        return {key:getattr(self, key) for key in keys}
    
    def copy(self):
        self.__class__(**self.get_params())
        
    @property
    def X(self): return self._X

class PreprocessMixin:
    def fit(self, X):
        self._X = self.parse_input(X)
        return self
    @property
    def X(self): return self._X
    @X.setter
    def X(self, what): self._X = self.parse_input(what)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
