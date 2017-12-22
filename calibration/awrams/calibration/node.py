# Worker with partial catchments, local objf

import multiprocessing as mp
try:
    mp.set_start_method('forkserver')
    p = mp.Process()
    p.start()
    p.join()
    print("n")
    #pass
except:
    pass


import numpy as np
from awrams.utils.messaging import Message
from awrams.utils.messaging.buffers import *
from awrams.calibration.allocation import allocate_cells_to_workers
from awrams.calibration.worker import WorkerProcess
from awrams.calibration.objective import ObjectiveProcess
from awrams.utils.nodegraph.nodes import funcspec_to_callable
from awrams.utils.nodegraph.graph import get_dataspecs_from_mapping
import time

 

class CatchmentSpec:
    def __init__(self,catchment_id,owns_results,partial_remote,communicator=None,np_buf=None,\
        shm_buf=None,split_counts=None,split_offsets=None):
        self.catchment_id = catchment_id
        self.owns_results = owns_results
        self.partial_remote = partial_remote
        self.communicator = communicator
        self.np_buf = np_buf
        self.shm_buf = shm_buf
        self.split_counts = split_counts
        self.split_offsets = split_offsets

    def __repr__(self):
        return "%s, %s" % (self.catchment_id, self.owns_results)


class CalibrationNode:

    def __init__(self):
        pass

    def run_node(self):
        '''
        Set up objectives, start objf process
        '''

        '''
        Set up MPI and receive init; receive allocation info, set up communicators etc
        Catchment list will be built from allocation info, then have shared buffers attached where need-be
        '''

        try:
            from mpi4py import MPI

            comm_world = MPI.COMM_WORLD

            init_msg = comm_world.recv(source=0)

            node_alloc = init_msg['node_alloc']
            catch_node_map = init_msg['catch_node_map']
            extent_map_full = init_msg['extent_map_full']
            run_period = init_msg['run_period']
            n_sub_workers = init_msg['n_sub_workers']
            owning_incl = init_msg['owning_incl']
            self.objective_spec = init_msg['objective_spec']
            objf_split_counts = init_msg['objf_split_counts']
            objf_split_offsets = init_msg['objf_split_offsets']
            node_mapping = init_msg['node_mapping']
            model = init_msg['model']
            #model_options = init_msg['model_options']

            self.model_outputs = self.objective_spec.catchment_objf.inputs_from_model
            self.local_objf_outputs = self.objective_spec.catchment_objf.output_schema

            '''
            non_worker_exclusions; list of global ranks which are not worker nodes or root
            extent_map_full; map catchment_ids to areas/masks (need to supply to workers for run_graph), plus aggregation
            run_period; period for which all cells are executed
            node_alloc; which catchments/cells belong to this node (incl ownership etc)
            catch_node_map; which nodes are attached to which catchments (determine communicator groups)
            n_sub_workers; number of sub_workers that this node should spawn
            schema
            '''

            '''
            Build objective spec
            '''

            g_world = comm_world.Get_group() 
            g_control_workers = g_world.Excl(init_msg['non_worker_exclusions'])
            comm_control_workers = comm_world.Create_group(g_control_workers)

            g_workers = g_control_workers.Excl([0])

            node_rank = g_workers.rank

            from awrams.calibration.allocation import build_node_splits

            ''' 
            Build catchmentspec dict;  data_index_map, communicators
            '''
            self.catchments = []

            data_index_map = build_data_index_map(node_alloc[node_rank]['catchments'])

            catchment_bufs = {}

            for cid,cell_info,ownership in node_alloc[node_rank]['catchments']:
                cspec = CatchmentSpec(cid,ownership['owns'],ownership['remote'])
                if cspec.partial_remote:
                    g_catch = g_workers.Incl(catch_node_map[cid])
                    cspec.communicator = comm_world.Create_group(g_catch)
                    cspec.split_counts,cspec.split_offsets = build_node_splits(node_alloc,catch_node_map,cid,len(run_period))
                    if cspec.owns_results:
                        data_shape = [extent_map_full[cid].cell_count,len(run_period)]
                        cspec.shm_buf = create_shm_dict(self.model_outputs,data_shape)
                        cspec.np_buf = shm_to_nd_dict(**cspec.shm_buf)
                        # Update buffer manager with these
                        catchment_bufs[cid] = cspec.shm_buf
                        #data_index_map_objf[cid] = None # We own the whole buffer, therefore there is no indexing

                self.catchments.append(cspec)

            self.owned_catchments = [c.catchment_id for c in self.catchments if c.owns_results]
            self.n_owned_catch = len(self.owned_catchments)
            self.catch_obj_index = dict([(c,i) for i,c in enumerate(self.owned_catchments)])

            if self.n_owned_catch > 0:
                g_control_owning = g_world.Incl(owning_incl)
                comm_control_owning = comm_world.Create_group(g_control_owning)


            worker_alloc = allocate_cells_to_workers(node_alloc[node_rank],extent_map_full,n_sub_workers)

            n_sub_workers = len(worker_alloc)

            '''
            Build shared memory buffers for workers/objective
            '''
            total_cells = sum(node_alloc[node_rank]['cell_counts'])
            data_shape = [len(run_period),total_cells]

            NBUFFERS = 4 # +++ Hardcode for now, just to get it working

            self.mcm = create_multiclient_manager(self.model_outputs,data_shape,NBUFFERS,n_sub_workers,True,catchment_bufs)

            '''
            Set up objective process
            '''
            obj_buf_handler = self.mcm.get_handler()

            self.objective = ObjectiveProcess(obj_buf_handler,data_index_map,extent_map_full,run_period,self.objective_spec,self.owned_catchments)
            self.objective.daemon = True

            self.objective.start()

            '''
            Set up model before sending to shared workers
            '''
            #model.set_outputs(schema['model_outputs'])
            #model.set_model_options()
            model.init_shared(get_dataspecs_from_mapping(node_mapping,True))


            '''
            Set up worker nodes; Shared buffers created from allocation info and general node settings
            '''
            

            client_bufs = self.mcm.get_client_managers()

            self.workers = [WorkerProcess(client_bufs[i],worker_alloc[i],extent_map_full,node_mapping,model,self.model_outputs,run_period) for i in range(n_sub_workers)]

            for w in self.workers:
                w.daemon = True
                w.start()

        

            start = None
            while(True):



                task = comm_control_workers.bcast(None,root=0)

                if start is None:
                    start = time.time()
                
                if task.subject == 'terminate':
                    self.terminate()
                    end=time.time()
                    print(end-start)
                    return

                self.submit_task(task)

                for i in range(task['njobs']):
                    task_buffer_id = self.get_results() # poll worker queues for job completion

                    # All below should be split into a post-processing stage (ie separate function)
                    # Right now it is hardcoded for lumped multi-catchment calibration
                    queued_objf = []
                    queued_send = []

                    for c in self.catchments: # +++ Sort by <owns_no_partial,owns_partial,other_owns>
                        if c.owns_results: # responsible for sending back to global
                            if c.partial_remote: # needs data from others

                                comm_c = c.communicator

                                r = []
                                out_tmp = []
                                data_local = self.mcm.map_buffer(task_buffer_id,data_index_map[c.catchment_id])

                                for s in self.model_outputs:

                                    data_local_s = np.ascontiguousarray(data_local[s].T)

                                    recv_buf = c.np_buf[s]

                                    r.append(comm_c.Igatherv(data_local_s,[recv_buf,c.split_counts,c.split_offsets,MPI.DOUBLE],root=0)) # needs to collect to shared buffer; but no buffer rotation required (1 buffer per catchment per variable)
                                    out_tmp.append((data_local_s,recv_buf))
                                    

                                queued_objf.append([r,c.catchment_id,c.catchment_id,out_tmp])

                                
                            else:
                                self.submit_objf(c.catchment_id,task_buffer_id,data_index_map[c.catchment_id],False)
                        else: # needs to send data to others
                            #+++ Should aggregate before sending for purely lumped catchments
                            # Do aggregation here or in objf?
                            comm_c = c.communicator
                            
                            r = []
                            out_tmp = []
                            

                            data = self.mcm.map_buffer(task_buffer_id,data_index_map[c.catchment_id])

                            for s in self.model_outputs:
                                out_data = np.ascontiguousarray(data[s].T)
                                r.append(comm_c.Igatherv(out_data,[None,c.split_counts,c.split_offsets,MPI.DOUBLE],root=0))
                                out_tmp.append(out_data)

                            queued_send.append([r,out_tmp])

                    for r,catchment_id,buffer_id,out_data in queued_objf:
                        for _r in r:
                            _r.wait()

                        self.submit_objf(catchment_id,buffer_id,None,True)


                    for r,out_data in queued_send:
                        for _r in r:
                            _r.wait()

                    objf_vals = self.get_objective()

                    self.mcm.reclaim(task_buffer_id)

                    if self.n_owned_catch > 0:
                        r = comm_control_owning.Igatherv(objf_vals,[None,objf_split_counts,objf_split_offsets,MPI.DOUBLE],root=0)
                        #comm_control_owning.Gatherv(objf_vals,[None,objf_split_counts,objf_split_offsets,MPI.DOUBLE],root=0)
                        r.wait()


        except Exception as e:
            print("Exception in node")
            print(e)
            comm_world.send(e,0,tag=999)
            self.terminate()

    def terminate(self):
        self.submit_task(Message('terminate'))
        for w in self.workers:
            w.join()
        self.objective.node_to_objective_q.put(Message('terminate'))
        self.objective.join()

    def submit_task(self,task):
        for w in self.workers:
            w.node_to_worker_q.put(task)

    def get_results(self):
        for w in self.workers:
            buf_id = w.worker_to_node_q.get()
        
        return buf_id

    def submit_objf(self,catchment_id,buffer_id,data_index=None,translate=False):
        self.objective.node_to_objective_q.put(Message('evaluate',(catchment_id,buffer_id,data_index,translate)))

    def get_objective(self):
        out_data = np.zeros((self.n_owned_catch,len(self.local_objf_outputs)))
        
        for i in range(self.n_owned_catch):
            catch,result = self.objective.objective_to_node_q.get()
            out_data[self.catch_obj_index[catch]] = result
            

        return out_data

def build_data_index_map(catchments):
    data_index_map = {}
    cur_cell = 0
    for c in catchments:
        next_cell = cur_cell + c[1]['ncells']
        data_index_map[c[0]] = np.s_[:,cur_cell:next_cell]
        cur_cell = next_cell
    return data_index_map

if __name__ == '__main__':
    node = CalibrationNode()
    node.run_node()


