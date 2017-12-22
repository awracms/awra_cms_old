'''
Declare the task specification for running SOBOL sampling of the AWRA-L model over a selection of catchments
'''

from awrams.calibration.support import *

def get_task_specification():

    from os.path import join
    from awrams.utils import gis
    from awrams.utils import datetools as dt
    from awrams.utils import extents

    from awrams.utils.settings_manager import get_settings

    paths = get_settings('data_paths')

    cal_extents = join(paths.CATCH_PATH,'calibration_extents_5k.nc')
    nces = gis.ExtentStoreNC(cal_extents,'r')

    catch_list_fn = join(paths.CATCH_PATH,'Catchment_IDs.csv')

    catch_df = open(catch_list_fn).readlines()
    cids = [k.strip().zfill(6) for k in catch_df[1:]]


    extent_map = {}
    extent_map = dict([(e,nces[e]) for e in cids])

    print([e.cell_count for e in extent_map.values()])

    run_period = dt.dates('1950 - 2011')
    eval_period = dt.dates('1970 - 2011')

    from awrams.calibration.sensitivity import SobolOptimizer

    #evolver_spec = EvolverSpec(sce.CCEvolver,evolver_run_args=dict(n_offspring=1,n_evolutions=5,elitism=2.0))

    #optimizer_spec = OptimizerSpec(sce.ShuffledOptimizer,evolver_spec=evolver_spec,n_complexes=14,max_nsni=1e100) #n_complex 14

    optimizer_spec = OptimizerSpec(SobolOptimizer,threshold = 0.005,max_eval =25000)

    from awrams.utils.nodegraph.nodes import callable_to_funcspec

    from awrams.calibration.objectives import test_objectives as tobj
    #local_objfspec = ObjectiveFunctionSpec(tobj.TestLocalSingle)
    #global_objfspec = callable_to_funcspec(tobj.TestGlobalSingle)

    observations=dict(qtot=join(paths.OBS_PATH,'qobs_zfill.csv'))

    local_objfspec = ObjectiveFunctionSpec(tobj.LocalQTotal)
    global_objfspec = callable_to_funcspec(tobj.GlobalQTotal)
    objective_spec = ObjectiveSpec(global_objfspec,local_objfspec,observations,eval_period)

    from awrams.models import awral
    from awrams.utils.nodegraph import nodes, graph
    node_mapping = awral.get_default_mapping()['mapping']
    model = callable_to_funcspec(awral)

    from awrams.models.settings import CLIMATOLOGY

    node_mapping['solar_clim'] = nodes.monthly_climatology(*CLIMATOLOGY['solar'])
    node_mapping['solar_filled'] = nodes.gap_filler('solar_f','solar_clim')
    node_mapping['rgt'].inputs[0] = 'solar_filled'

    '''
    User specifiable calibration description
    '''
    cal_spec = {}
    cal_spec['optimizer_spec'] = optimizer_spec
    cal_spec['objective_spec'] = objective_spec
    cal_spec['extent_map'] = extent_map
    cal_spec['run_period'] = run_period
    cal_spec['model'] = model
    cal_spec['node_mapping'] = node_mapping
    cal_spec['logfile'] = '/short/er4/dss548/test_sens.h5'

    cal_spec['prerun_action'] = callable_to_funcspec(prerun_raijin)
    cal_spec['postrun_action'] = callable_to_funcspec(postrun_raijin)

    return cal_spec

if __name__ == '__main__':
    cal_spec = get_task_specification()

    import sys

    max_nodes = int(sys.argv[1])
    walltime = sys.argv[2]

    cal_spec['logfile'] = '/short/er4/dss548/sobol_%s.h5' % max_nodes

    from awrams.calibration import cluster
    cluster.build_pbs_from_cal_spec(cal_spec,walltime,'/short/er4/dss548/sobol_%s.pkl'%max_nodes,'/short/er4/dss548/sobol_%s.pbs'%max_nodes,max_nodes)
 