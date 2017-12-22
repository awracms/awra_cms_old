'''
Settings related to calibration tasks

To override, create and edit ~/.awrams/calibration.py
'''

from os.path import dirname as _dirname
from os.path import join as _join

PROJECT = 'er4' # Default project for PBS jobs +++ BoM/WRM raijin specific
JOB_QUEUE = 'normal' # Default job queue for PBS jobs
CORES_PER_NODE = 16 # Number of cores per compute node
MEM_PER_NODE = 32,'gb' # Memory per compute note, units
ACTIVATION = ''

from awrams.utils.settings_manager import package_get_settings as _get_settings
exec(_get_settings('calibration'))
