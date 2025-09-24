# LEGEND SiPM Calibration

## Installation

Use  
`uv pip install -e ".[jupyter]"`

You can omit `[jupyter]` if you won't use notebooks.
Also, you can omit `uv` if you don't have it.  
Remember to set up a virtual environment before and activate it.

### On NERSC

When you are in a legendsw container e.g. on NERSC you can make use of all packages already installed there.
Create the virtual environment with  
`python -m virtualenv --system-site-packages .venv`  
in this case.
Then continue installing this package as described before.

## Usage

Either use a jupyter notebook (use the ipynb/autocalib.py) as a starting point or the executables.