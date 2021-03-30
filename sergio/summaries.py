'''
Created on Feb 8, 2021

@author: janis
'''

from colito.summaries import SummaryVisitor, SummaryState
from colito.logging import getModuleLogger
from sergio.language import Selector

log = getModuleLogger(__name__)

class SelectorSummariser(SummaryVisitor):
    def __init__(self, scores):
        self._scores = scores
    
    def on_summarised(self, state:SummaryState, actions):
        if isinstance(state.instance, Selector):
            selector = state.instance
            is_cache_enabled = selector.cache.enabled
            selector.cache.enabled = False
            value = state.value
            value['scores'] = self._evaluate_scores(selector)
            value['cached'] = {k:v for k,v in selector.cache.items()
                                 if isinstance(v,float)}
            selector.cache.enabled = is_cache_enabled
            #if summary_state.parts & SummaryParts.SELECTOR_VALIDITIES:
            #    records['validity'] = ValidityCodec.encode(selector.validity)
    
    def _evaluate_scores(self, selector):
        score_vals = {}
        for score in self._scores:
            try:
                tag = score.__collection_tag__
                val = score.evaluate(selector)
            except Exception as e:
                log.error(f'While evaluating {score}: {e}')
                val = float('nan')
            score_vals[tag] = val
        return score_vals
