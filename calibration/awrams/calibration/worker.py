import multiprocessing as mp
import numpy as np

from awrams.utils.nodegraph import nodes, graph
from awrams.utils import extents
from awrams.utils.mapping_types import gen_coordset
from awrams.calibration.support import get_params_from_mapping,flat_params_to_dict

class WorkerProcess(mp.Process):

    def __init__(self,buffer_manager,alloc_info,extent_map,mapping,model,model_outputs,run_period):
        super().__init__()

        self.buffer_manager = buffer_manager
        self.node_to_worker_q = mp.Queue()
        self.worker_to_node_q = mp.Queue()
        self.run_period = run_period
        self.alloc_info = alloc_info
        self.data_index = np.s_[:,alloc_info['offset']:alloc_info['offset']+alloc_info['size']]

        self.model_outputs = model_outputs

        self.mapping = mapping
        self.model = model
        self.coords, self.masks = build_multi_coords(alloc_info,extent_map,run_period)
        self.cell_count = alloc_info['size']
        self.timesteps = len(run_period)

        self.pspace = get_params_from_mapping(mapping)


    def build_static_graph(self,model_keys):
        parameterised, fixed_mapping,fixed_endpoints = graph.split_parameterised_mapping(self.mapping,model_keys)
        efixed = graph.ExecutionGraph(fixed_mapping)

        res = efixed.get_data_flat(self.coords,self.masks,multi=True)
        fixed_data = dict((k,res[k]) for k in fixed_endpoints)

        dspecs = efixed.get_dataspecs()

        fixed_mapping = expand_const(fixed_mapping)

        static_mapping = dict([k,nodes.static(fixed_data[k],dspecs[k],fixed_mapping[k].out_type)] for k in fixed_endpoints)
        static_mapping.update(parameterised)

        estatic = graph.ExecutionGraph(static_mapping)

        return estatic


    def run(self):
        import os
        all_cores = set(range(mp.cpu_count()))

        os.sched_setaffinity(0,all_cores)

        self.buffer_manager.rebuild_buffers()

        #self.model = nodes.funcspec_to_callable(self.model)

        '''
        Build the extent map
        '''


        '''
        Initialise our model/graphrunner
        Need to do a first run of fixed parameter 
        '''

        model_keys = self.model.get_input_keys()
        self.exe_graph = self.build_static_graph(model_keys)

        self.model_runner = self.model.get_runner(self.exe_graph.get_dataspecs(True),shared=False)

        while(True):
            task = self.node_to_worker_q.get()

            if task.subject == 'terminate':
                return

            for i in range(task['njobs']):
                #get_buffers()

                #results = run_graph(task_params)
                buf_id, data = self.buffer_manager.get_buffer(index=self.data_index)

                #for k,v in data.items():
                #    v[...] = i * buf_id
                job = task['jobs'][i]

                pdict = flat_params_to_dict(job['params'],self.pspace)

                self.exe_graph.set_parameters(pdict)
                iresults = self.exe_graph.get_data_flat(None,0) # +++ refactor get_data_prepack
                mresults = self.model_runner.run_from_mapping(iresults,self.timesteps,self.cell_count)

                for k in self.model_outputs:
                    data[k][...] = mresults[k]

                self.worker_to_node_q.put(buf_id)

def build_multi_coords(alloc_info,extent_map,period):
    
    esubs = []
    
    for c in alloc_info['catchments']:
        esubs.append(extents.split_extent(extent_map[c['cid']],c['ncells'],c['start_cell']))
    
    coords = []
    masks = []
          
    for e in esubs:
        coords.append(gen_coordset(period,e))
        masks.append(e.mask)
        
    return coords,masks

from numbers import Number

def expand_const(mapping): #+++ Should relocate to nodegraph
    mapping = mapping.copy()
    const_req = set()
    for v in mapping.values():
        for i in v.inputs:
            if isinstance(i,Number):
                const_req.add(i)
    
    for i in const_req:
        if i not in mapping:
            mapping[i] = nodes.const(i)
    return mapping