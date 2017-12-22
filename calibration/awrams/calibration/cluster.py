import os
import sys
from awrams.calibration import settings
from awrams.calibration.allocation import allocate_catchments_to_nodes
from awrams.utils.nodegraph.nodes import callable_to_funcspec

def build_pbs_from_cal_spec(cal_spec,walltime,pickle_fn,pbs_fn,\
    max_nodes,core_min=1,max_over=1.02,job_name='awrams_cal',job_queue=None,project=None):
    
    full_spec = build_pickle_from_spec(cal_spec,max_nodes,pickle_fn,core_min,max_over)
    n_nodes = full_spec['n_workers']
    build_pbs_file(n_nodes,walltime,pickle_fn,pbs_fn,job_name,job_queue,project)


def build_pickle_from_spec(cal_spec,max_nodes,pickle_fn,core_min=1,max_over=1.02):
    
    n_cores = settings.CORES_PER_NODE * core_min
    node_alloc,catch_node_map = allocate_catchments_to_nodes(cal_spec['extent_map'],max_nodes,n_cores,max_over=max_over)

    n_workers = len(node_alloc)

    cal_spec = cal_spec.copy()

    for k in ['prerun_action','postrun_action']:
        calldef = cal_spec.get(k)
        if calldef is not None:
            if isinstance(calldef,dict):
                pass
            else:
                cal_spec[k] = callable_to_funcspec(calldef)

    cal_spec['node_alloc'] = node_alloc
    cal_spec['catch_node_map'] = catch_node_map
    cal_spec['n_workers'] = n_workers

    import pickle

    with open(pickle_fn,'wb') as pkl_out:
        pickle.dump(cal_spec,pkl_out)

    return cal_spec


def build_pbs_file(n_nodes,walltime,pickle_fn,pbs_fn,job_name='awrams_cal',job_queue=None,project=None):

    fh = open(pbs_fn,'w')

    n_cores = int(n_nodes)*settings.CORES_PER_NODE
    mem_scale,mem_units = settings.MEM_PER_NODE
    mem_string = str(int(n_nodes*mem_scale))+mem_units

    if job_queue is None:
        job_queue = settings.JOB_QUEUE
    if project is None:
        project = settings.PROJECT

    activation = settings.ACTIVATION

    base_script = \
    """
    #!/bin/bash
    #PBS -q {job_queue}
    #PBS -P {project}
    #PBS -N {job_name}
    #PBS -l walltime={walltime}
    #PBS -l ncpus={n_cores}
    #PBS -l mem={mem_string}

    {activation}

    python3 -m awrams.calibration.launch_calibration {pickle_fn}

    """.format(**locals())

    fh.write(base_script)
    fh.close()

if __name__ == '__main__':
    base_path = os.environ['AWRAPATH']
    calpkl = 'calibration.pkl'
        
    usage="""
    Usage: NUM_NODES WALLTIME
    Example: 16 01:30:00

    NUM_NODES   Number of nodes needed for the run.
    WALLTIME    Duration of the run, format hh:mm:ss.
    """

    if len(sys.argv) != 3:
        print(usage)
        sys.exit()

    nodes = sys.argv[1]
    walltime = sys.argv[2]

    build_pbs_file(nodes,walltime,base_path,calpklfilename)