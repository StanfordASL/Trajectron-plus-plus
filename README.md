# Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control #

This repository contains the code for [Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control](https://arxiv.org/abs/2001.03093) by Tim Salzmann\*, Boris Ivanovic\*, Punarjay Chakravarty, and Marco Pavone (\* denotes equal contribution).

Specifically, this branch is for the Trajectron++ applied to the nuScenes autonomous driving dataset.

## Installation ##

### Note about Submodules ###
When cloning this branch, make sure you clone the submodules as well, with the following command:
```
git clone --recurse-submodules <repository cloning URL>
```
Alternatively, you can clone the repository as normal and then load submodules later with:
```
git submodule init # Initializing our local configuration file
git submodule update # Fetching all of the data from the submodules at the specified commits
```

### Environment Setup ###

First, we'll create a conda environment to hold the dependencies.
```
conda create --name trajectron++ python=3.6 -y
source activate trajectron++
pip install -r requirements.txt
```

Then, since this project uses IPython notebooks, we'll install this conda environment as a kernel.
```
python -m ipykernel install --user --name trajectron++ --display-name "Python 3.6 (Trajectron++)"
```

Now, you can start a Jupyter session and view/run all the notebooks in `code/notebooks` with
```
jupyter notebook
```

When you're done, don't forget to deactivate the conda environment with
```
source deactivate
```

## Scripts ##

Run any of these with a `-h` or `--help` flag to see all available command arguments.
* `code/train.py` - Trains a new Trajectron++ model.
* `code/notebooks/run_eval.bash` - Evaluates the performance of the Trajectron++. This script mainly collects evaluation data, which can then be visualized with `code/notebooks/NuScenes Quantitative.ipynb`.
* `data/nuScenes/process_nuScenes.py` - Processes the nuScenes dataset into a format that the Trajectron++ can directly work with, following our internal structures for handling data (see `code/data` for more information).
* `code/notebooks/NuScenes Qualitative.ipynb` - Visualizes the predictions that the Trajectron++ makes.

## Datasets ##

A sample of fully-processed scenes from the nuScenes dataset are available in this repository, in `data/processed`.

If you want the *original* nuScenes dataset, you can find it here: [nuScenes Dataset](https://www.nuscenes.org/).
