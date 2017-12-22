# AWRA Community Modelling System


## Installation

### Python Environment Dependencies

The modelling system has been developed, tested and used in Linux environments. Installation has been tested on several flavours of Linux (Ubuntu, Fedora, Mint and “Bash on Ubuntu on Windows”).

The modelling system is designed to be run from Python 3 (specifically has been developed and tested with v3.4.5) and relies on the following capabilities

* C compiler to build the core model code
* NetCDF library (version 4.3, NOTE: 4.4 creates files that are unreadable with h5py) and Python bindings (NetCDF4)
* HDF5 library (Tested on 1.8 and up) and Python bindings (h5py)
* Python numpy and pandas packages
* IPython/Jupyter notebook for the recommended interactive use of the modelling system
* and various Python packages including:
  * `matplotlib` for image and graph display
  * `nose` for running tests
  * `cffi` for building the model Python bindings
  * `pyzmq` for inter-process communication

Certain functions also rely on the osgeo/GDAL libraries and corresponding Python bindings. To use shape files GDAL needs to be installed. However, this is optional, as the simulation package will work without it.

The following sections give guidance for setting up a conda environment and installing the AWRA-CMS.


tested on 4/10/2017

#### download the AWRA-CMS
```
# download a zip file from github
wget https://github.com/awracms/awra_cms/archive/master.zip
unzip master.zip
cd awra_cms-master
# OR download a tarball from github
wget https://github.com/awracms/awra_cms/archive/master.tar.gz
tar zxf master.tar.gz
cd awra_cms-master
# OR clone the repository on github
git clone https://github.com/awracms/awra_cms.git
cd awra_cms
```

#### install conda and build the AWRA-CMS conda environment
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh

conda create -n awra-cms --file conda_install_env.lst
```

#### to activate the AWRA-CMS conda environment
```
source activate awra-cms
```

#### install AWRA-CMS into conda environment
```
### install-awra-cms.sh
cd utils
pip install .

cd ../benchmarking
pip install .
python setup.py nosetests

cd ../models
pip install .
python setup.py nosetests

cd ../simulation
pip install .
python setup.py nosetests

cd ../visualisation
pip install .
python setup.py nosetests

cd ../calibration
pip install .
python setup.py nosetests

cd ../utils
python setup.py nosetests
cd ..
```


#### Bash on Ubuntu on Windows with the Windows SubSystem for Linux (WSL)

AWRA-CMS has been successfully tested on WSL

WSL will run on a 64-bit version of Windows 10 Anniversary Update build 14393 or later

To enable WSL, follow the instructions at:  [https://msdn.microsoft.com/en-au/commandline/wsl/install_guide](https://msdn.microsoft.com/en-au/commandline/wsl/install_guide)

Then open a command prompt and type *bash* and complete the installation process as above

To start notebook server, type at the bash command prompt
```
jupyter notebook --no-browser
```
then open a browser and point to http://localhost:8888/notebooks

Nosetests will only run python test files with all execution permissions turned off *chmod -x test_file.py*
