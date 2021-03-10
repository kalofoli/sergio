'''
Created on Jan 31, 2021

@author: janis
'''

import argparse
import traceback

from colito.logging import getModuleLogger, setAdapterFactory, SergioLogger
from sergio.predicates import DEFAULT_PREDICISER
setAdapterFactory(SergioLogger)
log = getModuleLogger(__name__)


from colito.config import ActionParser

from sergio.computation import Computation
from sergio.config import ExperimentActions






if __name__ == '__main__':
    log.add_stderr()
    log.setLevel('INFO')

    from sergio.data.factory import DatasetFactory
    from sergio import FileManager
    fm = FileManager(paths={'data':'../../data/'})
    ds = DatasetFactory(file_manager=fm, cache=None).load_dataset('twitter')
    preds = tuple(ds.make_predicates(DEFAULT_PREDICISER))
    
    print(preds)
    
if __name__ == 'z__main__':
    import sys
    log.add_stderr()
    
    experiment = Computation()
    ap = argparse.ArgumentParser(prog=__package__)
    ap.add_argument('--dry-run', dest='dry_run', default=False, action='store_true',help='Do not perform any action. Just validate the inputs.')
    action_parser = ActionParser(__package__, experiment, parser=ap)

    argv = ['sdcore',
            'config', '-R', '-l', 'PROGRESS', '-t','pydebug','-D', 'INFO' ,'0','-D','PROGRESS','2', '-p','LOG','output/logs/','-p','WORK','output/work',
            'control','-s','SIGUSR1','SUMMARISE_JSON','-o','/tmp/{config.tag_formatted}.json',
            #'load-dataset', '-n', 'petsters:cats',
            # 'load-dataset', '-n', 'imdb:bool',
            'load-dataset', '-n', 'ego:facebook,tags=10','-N','BOOLEAN|RANGED','-r','SLABS_POSITIVE','-c','3',
            #'load-dataset', '-n', 'gattwto:1999',
            # 'load-language', '-n', 'closure-conjunctions-slow',
            'load-language', '-n', 'closure-conjunctions-restricted',
            # 'load-subgroups','-s','{abs:efficiency^abs:linear}','{abs:linear^abs:process}','{abs:control^abs:linear}','--',
            # 'load-scores', '-m', 'local-modularity', '-e', 'local-modularity', # ,'-g','.2',
            'load-scores', '-m', 'coverage-average-coreness', '-e', 'coverage-induced-max-coreness', '-g','.1',
            #'load-scores', '-m', 'inverse-conductance', '-e', 'inverse-conductance', '-s','.01',
            'optimise', '-d', '5','-a','1','-S','coverage:increasing=True','-R','selector:record=indices','-k','1','-t',
            'summarise-json','-i','2','-o','/tmp/out-all.json','-p','PREDICATE_INDICES|SELECTOR_VALIDITIES|LANGUAGE_PREDICATES','-O',
            #'output-json', '-a', '-o', '/tmp/out.txt','-s',',',':',
            ]
    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p DATA testing/data -t testing".split()  +
            "load-dataset -n delicious:tags=10,mode=num -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-language -n conjunctions".split() +
            "export-dataset --output {load-dataset.name}.mat".split())

    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p DATA testing/data -t testing".split()  +
            "load-dataset -n delicious:tags=10,mode=num -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-language -n conjunctions".split() +
            "load-attributes -gf matlab/results/Delicious:30:bool-pics.csv.gz -s false -d bool".split() +
            "load-scores -m jaccard -e jaccard -C target pics_1 ".split() +
            "export-dataset -O --output {load-dataset.name}-ext.mat".split())

    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p WORK testing -D PROGRESS 6".split()  +
            "-t delicious_tags=10,mode=num_4_0-a1-evr-{runtime.hostname} control -s SIGUSR1 SUMMARISE_JSON".split() +
            "-f -o output/running/{config.tag_formatted}.json load-dataset -n delicious:tags=10,mode=num load-language".split()+
            "-n closure-conjunctions-restricted load-scores -m edge-vertex-ratio -e edge-vertex-ratio-greedy optimise -d 4 -k 1".split() +
            "-S value_addition:mult_fval=4,mult_oest=1 -R optimistic_estimate:increasing=True -a 1.0 summarise-json -p ALL -i 2".split() +
            "-o test.json -O".split())
    
    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p DATA testing/data -t testing".split()  +
            "load-dataset -n delicious:tags=10,mode=num -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-language -n conjunctions".split() +
            "load-attributes -f matlab/results/delicious:tags=50,mode=bool-testing.csv -s false -d bool".split() +
            "load-scores -m geometric-mean -e geometric-mean -C exponents".split() + ["8 1"] + ['-M','measures','jaccard(pics_1) coverage-average-coreness(gamma=.5)',
                                                                                                '-E', 'optimistic_estimators', 'jaccard(pics_1) coverage-induced-average-coreness(gamma=.5)'] +
            "optimise -d 3 -k 1 -S value_addition:mult_fval=4,mult_oest=1 -R selector:record=indices -a 1 -t".split() +
            "summarise-json -p ALL -i 2 -o 'output/results/{config.tag_formatted}.json'".split())
            
    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p DATA testing/data -t testing".split()  +
            "load-dataset -n ego:facebook,tags=20 -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-language -n conjunctions".split() +
            "export-dataset -O --output {load-dataset.name}.mat".split())

    argv = ("sdcore config -Rl PROGRESS -p LOG testing -p DATA testing/data -t testing".split()  +
            #"load-dataset -n delicious:tags=10,mode=num -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-dataset -n lastfm-hetrec:15,bool -N BOOLEAN|RANGED -r SLABS_POSITIVE -c 3".split() +
            "load-language -n conjunctions".split() +
            "communities -k 2".split() +
            "summarise-json -Op PREDICATE_INDICES -i 2 -o output/testing/{config.tag_formatted}.json".split())
    
    argv = ('../sdcore/sdcore.py config -Rl PROGRESS -p LOG testing/ -p WORK output/results -D PROGRESS 30 -t'.split() +
            'lastfm-hetrec:15,bool-5-picked_opt_galbrun_5-{runtime.hostname} control -s SIGUSR1 SUMMARISE_JSON -f -o output/running/{config.tag_formatted}.json'.split() +
            'load-dataset -n lastfm-hetrec:15,bool -c 5 -N BOOLEAN|RANGED -r SLABS_POSITIVE'.split() +
            'load-language -n closure-conjunctions-restricted communities -k 5 summarise-json -p PREDICATE_INDICES|LANGUAGE_PREDICATES -i 2 -Oo testing/results/testing.json'.split())
    if not (len(sys.argv)==2 and sys.argv[1]=='--use-static-arguments'): 
        argv = sys.argv
    else:
        sys.stderr.write(f'Overriding command line arguments for debugging.')
    
    nmsps = []
    actions = action_parser.parse_arguments(argv[1:], namespaces=nmsps)
    actions = ExperimentActions(actions)
    actions.argv = argv
    try:
        for action in actions:
            action.validate()
        if not nmsps[0].dry_run:
            for action in actions:
                action.perform()
    except OSError as e:
        traceback.print_exc()
        sys.exit(e.errno)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
    # sdcore.cache.DefaultCache.disable()

