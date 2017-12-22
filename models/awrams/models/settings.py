'''
Settings related to model runs

To override, create and edit ~/.awrams/models.py
'''

from os.path import dirname as _dirname
from os.path import join as _join

from awrams.utils.settings import TRAINING_DATA_PATH

CLIMATE_DATA = _join(TRAINING_DATA_PATH,'climate/BOM_climate/')

FORCING = {
    'tmin': ('temp_min_day/temp_min*','temp_min_day'),
    'tmax': ('temp_max_day/temp_max*','temp_max_day'),
    'precip': ('rain_day/rain*','rain_day'),
    'solar': ('solar_exposure_day/solar*','solar_exposure_day')
}

CLIMATOLOGY_PATH = _join(TRAINING_DATA_PATH,'simulation/climatology')

CLIMATOLOGIES = {
	'solar': (_join(CLIMATOLOGY_PATH,'Rad_1990_2009.nc'),'solar_exposure_day')
}

from awrams.utils.settings_manager import package_get_settings as _get_settings
exec(_get_settings('models'))
