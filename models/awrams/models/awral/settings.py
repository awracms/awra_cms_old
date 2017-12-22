'''
Settings for AWRA-L model

To override, create and edit ~/.awrams/awral.py
'''

from os.path import dirname as _dirname
from os.path import join as _join

# Default calibrated parameters
DEFAULT_PARAMETER_FILE=_join(_dirname(__file__),'data/DefaultParameters.json')

# Default spatial gridded inputs
SPATIAL_FILE = _join(_dirname(__file__),'data/spatial_parameters.h5')

# List of input keys
INPUT_JSON_FILE=_join(_dirname(__file__),'data/model_inputs.json')

# Default path for initial states files
# Only used in training examples, should not be considered 'standard'

from awrams.utils.settings import TRAINING_DATA_PATH
INITIAL_STATES_PATH = _join(TRAINING_DATA_PATH,'simulation/initial_states/')

# Model outputs
DEFAULT_OUTPUTS = dict(
	OUTPUTS_HRU = ['s0','ss','sd','mleaf'], # Outputs from each HRU
    OUTPUTS_AVG = ['e0','etot','dd','s0','ss','sd'], # Outputs that are a weighted average of HRU values
    OUTPUTS_CELL = ['qtot','sr','sg'] # Cell level outputs
)

# Model build template
# This only needs to be fully populated for manual builds of the model, for dynamic builds this is
# generated at runtime

DEFAULT_TEMPLATE = dict(
    OUTPUTS_HRU = DEFAULT_OUTPUTS['OUTPUTS_HRU'],
    OUTPUTS_AVG = DEFAULT_OUTPUTS['OUTPUTS_AVG'],
    OUTPUTS_CELL = DEFAULT_OUTPUTS['OUTPUTS_CELL'],
    INPUTS_SCALAR = ['kr_coeff','slope_coeff','pair'],
    INPUTS_SCALAR_HRU = ['alb_dry', 'alb_wet', 'cgsmax', 'er_frac_ref', 'fsoilemax','lairef', 'rd', 'sla', 'vc', \
                  'w0ref_alb','us0', 'ud0', 'wslimu', 'wdlimu', 'w0lime','s_sls','tgrow','tsenc'],
    INPUTS_SPATIAL = ['k_rout','k_gw','k0sat','kssat','kdsat','kr_0s','kr_sd','s0max','ssmax','sdmax','prefr','slope'],
    INPUTS_SPATIAL_HRU = ['fhru','hveg','laimax'],
    INPUTS_FORCING = ['tat','pt','rgt','avpt','u2t','radcskyt']
)

ICC_BUILD_STR = "icc %s -std=c99 -static-intel --shared -fPIC -O3 -o %s"
GCC_BUILD_STR = "gcc %s -std=c99 --shared -fPIC -O3 -o %s"

# Set this to determine which compiler is used for dynamic compilation

BUILD_STR = GCC_BUILD_STR