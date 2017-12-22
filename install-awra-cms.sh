# source activate awra-cms

cd utils
# rm -r awrams.utils.egg-info
pip install --upgrade .

cd ../benchmarking
# rm -r awrams.benchmarking.egg-info
pip install --upgrade .
python setup.py nosetests

cd ../models
# rm -r awrams.models.awral.egg-info
pip install --upgrade .
python setup.py nosetests

cd ../simulation
# rm -r awrams.simulation.egg-info
pip install --upgrade .
python setup.py nosetests

cd ../visualisation
# rm -r awrams.visualisation.egg-info
pip install --upgrade .
python setup.py nosetests

cd ../calibration
# rm -r awrams.calibration.egg-info
pip install --upgrade .
python setup.py nosetests

cd ../utils
python setup.py nosetests
cd ..

