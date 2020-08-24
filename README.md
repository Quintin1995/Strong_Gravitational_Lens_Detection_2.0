# Strong Gravitational Lens Detection 2.0
This project is about detecting strong gravitational lenses in the Kilo Degree Survey.

## Based on
This project is based on the following project: https://github.com/CEnricoP/cnn_strong_lensing & paper: https://arxiv.org/abs/1702.07675

## Requirements
Use the following command to install required packages: 
~~~
pip3 install -r requirements.txt
~~~


## How to run
~~~
python3 main.py --run=runs/experiment_folder/run.yaml
~~~

Input parameters of a run can be set in the following file: runs/experiment_folder/run.yaml.

## Data
Data can be requested.

The data is split in the following way:
- Train Data      80%
- Validation Data 10%
- Test Data       10%

In this binary classification problem three types of images with dimensions (101,101,1) are used:
- 100000 Sources  (Simulated Lensing features as *.fits* files.)
- 5513 Lenses     (An image of a galaxy probalby not showing strong gravitational lensing features.)
- 6083 Negatives  (An image identified as not showing strong gravitational lensing features.)


More detailed information will be added later on. At this stage in the project, changes will be frequent.

