from .settings import DEFAULT_PARAMETER_FILE,SPATIAL_FILE
from awrams.models.settings import CLIMATE_DATA,FORCING
from awrams.models.model import Model

class AWRALModel(Model):

    def __init__(self,model_options=None):
        from .settings import DEFAULT_OUTPUTS
        self.OUTPUTS = DEFAULT_OUTPUTS

        self._SHARED = None
        
    def get_runner(self,dataspecs,shared=False):
        """
        Return a ModelRunner for this model
        
        Args:
            dataspecs (dict): Dataspecs (as returned from ExecutionGraph)
            shared (bool): Is this runner being used in a shared memory context?

        Returns:
            ModelRunner
        """
        from . import runner as fw
        if shared:
            if self._SHARED is not None:
                return fw.FFIWrapper(mhash=self._SHARED['mhash'],template=self._SHARED['template'])
            else:
                raise Exception("Call init_shared before using multiprocessing")
        else:
            template = fw.template_from_dataspecs(dataspecs,self.OUTPUTS)
            return fw.FFIWrapper(False,template)

    def init_shared(self,dataspecs):
        '''
        Call before attempting to use in multiprocessing
        '''
        from . import runner as fw
        template = fw.template_from_dataspecs(dataspecs,self.OUTPUTS)
        mhash = fw.validate_or_rebuild(template)

        self._SHARED = dict(mhash=mhash,template=template)

    def get_input_keys(self):
        """
        Return the list of keys required as inputs

        Returns:
            list

        """
        from .support import get_input_keys
        return get_input_keys()

    def get_state_keys(self):
        """
        Return the list of keys representing model states

        Returns:
            list
        """
        from .support import get_state_keys
        return get_state_keys()

    def get_output_variables(self):
        """
        Return the list of output variable keys for this model

        Returns:
            list
        """
        output_vars = []
        for v in self.OUTPUTS['OUTPUTS_AVG'] + self.OUTPUTS['OUTPUTS_CELL']:
            output_vars.append(v)
        for v in self.OUTPUTS['OUTPUTS_HRU']:
            output_vars.extend([v+'_hrusr',v+'_hrudr'])
        return output_vars

    def get_default_mapping(self):
        """
        Return the default input mapping for this model
        This is a dict of key:GraphNode mappings

        Return:
            mapping (dict)
        """
        import json
        from awrams.utils.nodegraph import graph, nodes
        from awrams.utils.metatypes import PrettyObjectDict
        from . import transforms
        import numpy as np

        dparams = json.load(open(DEFAULT_PARAMETER_FILE,'r'))
        #dparams = dict([(k.lower(),v) for k,v in dparams.items()])
        for entry in dparams:
            entry['MemberName'] = entry['MemberName'].lower()

        mapping = {}

    #    for k,v in dparams.items():
    #        mapping[k] = nodes.const(v)

        for entry in dparams:
            #tmp = entry.copy()
            #tmp.pop('MemberName')
            #tmp.pop('Value')
            mapping[entry['MemberName']] = nodes.parameter(entry['Value'],entry['Min'],entry['Max'],entry['Fixed'],description=entry['DisplayName'])
        # Setup a new-style functional input map

        import h5py
        ds = h5py.File(SPATIAL_FILE,mode='r')
        SPATIAL_GRIDS = list(ds['parameters'])
        ds.close()

        for k,v in FORCING.items():
            mapping[k+'_f'] = nodes.forcing_from_ncfiles(CLIMATE_DATA,v[0],v[1])
            
        for grid in SPATIAL_GRIDS:
            #if grid == 'height':
            #    mapping['height'] = nodes.hypso_from_hdf5(SPATIAL_FILE,'parameters/height')
            #else:
            mapping[grid.lower()+'_grid'] = nodes.spatial_from_file(SPATIAL_FILE,'parameters/%s' % grid)

        mapping.update({
            'tmin': nodes.transform(np.minimum,['tmin_f','tmax_f']),
            'tmax': nodes.transform(np.maximum,['tmin_f','tmax_f']),
            'hypsperc_f': nodes.const_from_hdf5(SPATIAL_FILE,'dimensions/hypsometric_percentile',['hypsometric_percentile']),
            'hypsperc': nodes.mul('hypsperc_f',0.01), # Model needs 0-1.0, file represents as 0-100
            'fday': transforms.fday(),
            'u2t': transforms.u2t('windspeed_grid','fday')
        })

        mapping['height'] = nodes.assign('height_grid')

        mapping['er_frac_ref_hrusr'] = nodes.mul('er_frac_ref_hrudr',0.5)

        mapping['k_rout'] = nodes.transform(transforms.k_rout,('k_rout_scale','k_rout_int','meanpet_grid'))
        mapping['k_gw'] = nodes.mul('k_gw_scale','k_gw_grid')

        mapping['s0max'] = nodes.mul('s0max_scale','s0fracawc_grid',100.)
        mapping['ssmax'] = nodes.mul('ssmax_scale','ssfracawc_grid',900.)
        mapping['sdmax'] = nodes.mul('ssmax_scale','sdmax_scale','ssfracawc_grid',5000.)

        mapping['k0sat'] = nodes.mul('k0sat_scale','k0sat_v5_grid')
        mapping['kssat'] = nodes.mul('kssat_scale','kssat_v5_grid')
        mapping['kdsat'] = nodes.mul('kdsat_scale','kdsat_v5_grid')

        mapping['kr_0s'] = nodes.transform(transforms.interlayer_k,('k0sat','kssat'))
        mapping['kr_sd'] = nodes.transform(transforms.interlayer_k,('kssat','kdsat'))

        mapping['prefr'] = nodes.mul('pref_gridscale','pref_grid')
        mapping['fhru_hrusr'] = nodes.sub(1.0,'f_tree_grid')
        mapping['fhru_hrudr'] = nodes.assign('f_tree_grid')
        mapping['ne'] = nodes.mul('ne_scale','ne_grid')
        mapping['slope'] = nodes.assign('slope_grid')
        mapping['hveg_hrudr'] = nodes.assign('hveg_dr_grid')

        mapping['laimax_hrusr'] = nodes.assign('lai_max_grid')
        mapping['laimax_hrudr'] = nodes.assign('lai_max_grid')

        mapping['pair'] = nodes.const(97500.)

        mapping['pt'] = nodes.assign('precip_f')
        mapping['rgt'] = nodes.transform(np.maximum,['solar_f',0.1])
        mapping['tat'] = nodes.mix('tmin','tmax',0.75)
        mapping['avpt'] = nodes.transform(transforms.pe,'tmin')
        mapping['radcskyt'] = transforms.radcskyt()

        mapping['init_sr'] = nodes.const(0.0)
        mapping['init_sg'] = nodes.const(100.0)
        for hru in ('_hrusr','_hrudr'):
            mapping['init_mleaf'+hru] = nodes.div(2.0,'sla'+hru)
            for state in ["s0","ss","sd"]:
                mapping['init_'+state+hru] = nodes.mul(state+'max',0.5)

        return PrettyObjectDict(mapping)

