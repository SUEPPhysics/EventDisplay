# EventDisplay
Simple event display scripts for SUEPs analysis

## Getting started
First, log into the LPC cluster (or any cluster with CVMFS access) and set up the environment
```bash
cmsrel CMSSW_11_1_4
cd CMSSW_11_1_4/src
cmsenv
```
Then, clone the repository and create a directory for saving the figures
```bash
git clone https://github.com/SUEPPhysics/EventDisplay.git
cd EventDisplay
mkdir Results
```
Install any necessary python modules
```bash
python3 -m pip install mplhep pyjet
```

**Note: You can also set this up locally by installing the rest of the necessary modules using the last command. However, you might have trouble accessing files stored in EOS using xrdcp.**

# Running the event display script
To create a simple event display figure execute
```bash
python3 scripts/eventDisplay.py
```
Most options can be set by modifying the input section in the beginning of the script.

# Running an event loop (and event shapes)
For the moment being, the repository contains two scripts that have event loops.
The first is 
```bash
python3 scripts/clustering.py
```
Its purpose is to study the results of clustering using pyjet.
The other script is 
```bash
python3 scripts/plotEventShapes.py
```
This script calculates and plots the distributions of various event shape variables.
