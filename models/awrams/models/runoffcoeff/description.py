from awrams.models.model import Model

class RunoffModel(Model):

    '''
    A simple single-coefficient linear runoff model
    This should be considered the 'reference model' for development
    of other parts of the codebase
    '''

    def get_input_keys(self):
        '''
        Return a list of strings representing the inputs
        to the model (ie the values that must be present in an ExecutionGraph
        in order for the model to run)
        '''
        return ['c','precip']

    def get_runner(self,dataspecs,shared=False):
        '''
        Return an object capable of running the model
        '''
        from .runner import RunoffRunner
        return RunoffRunner()

    def get_output_variables(self):
        return ['q']

    def get_default_mapping(self):
        '''
        Load the default (NodeGraph) mapping for this model
        '''
        from awrams.utils.nodegraph import nodes
        from awrams.models.settings import CLIMATE_DATA

        mapping = {}
        mapping['c'] = nodes.parameter(0.5,0.0,1.0,False)
        mapping['precip'] = nodes.forcing_from_ncfiles(CLIMATE_DATA,'rr_*','rain_day')
        return mapping

