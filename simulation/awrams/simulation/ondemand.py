from awrams.utils.nodegraph import graph,nodes
from awrams.utils import mapping_types as mt
from awrams.utils import extents

class OnDemandSimulator:
    def __init__(self,model,imapping,omapping=None,extent=None):

        if extent is None:
            extent = extents.get_default_extent()

        imapping = graph.get_input_tree(model.get_input_keys(),imapping)
        #+++
        #Document the use of this manually, don't just change the graph behind the scenes...
        #imapping = graph.map_rescaling_nodes(imapping,extent)

        self.input_runner = graph.ExecutionGraph(imapping)
        self.model_runner = model.get_runner(self.input_runner.get_dataspecs(True))

        self.outputs = None
        if omapping is not None:
            self.outputs = graph.OutputGraph(omapping)

    def run(self,period,extent,return_inputs=False):
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(period,extent)

        iresults = self.input_runner.get_data_flat(coords,extent.mask)
        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        if return_inputs:
            return mresults,iresults
        else:
            return mresults

    def run_prepack(self,iresults,period,extent):
        '''
        run with pre-packaged inputs for calibration
        :param cid:
        :param period:
        :param extent:
        :return:
        '''
        coords = mt.gen_coordset(period,extent)
        if self.outputs:
            ### initialise output files if necessary
            self.outputs.initialise(period,extent)

        mresults = self.model_runner.run_from_mapping(iresults,coords.shape[0],extent.cell_count)

        if self.outputs is not None:
            self.outputs.set_data(coords,mresults,extent.mask)

        return mresults
