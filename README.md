**NOTE:** This branch is obsolete, it is only kept around for posterity.

# Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control #

This repository contains the code for [Trajectron++: Multi-Agent Generative Trajectory Forecasting With Heterogeneous Data for Control](https://arxiv.org/abs/2001.03093) by Tim Salzmann\*, Boris Ivanovic\*, Punarjay Chakravarty, and Marco Pavone (\* denotes equal contribution).

Specifically, this branch is for the Trajectron++ applied to the ETH and UCY pedestrian datasets. 

## Installation ##

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
* `code/notebooks/model_to_metric_csv.py` - Evaluates the performance of the Trajectron++. This script mainly collects evaluation data, which can be visualized with `code/notebooks/Result Analysis.ipynb`.
* `data/process_data.py` - Processes the ETH and UCY datasets into a format that the Trajectron++ can directly work with, following our internal structures for handling data (see `code/data` for more information).

## Datasets ##

Preprocessed ETH and UCY datasets are available in this repository, under `data/` folders (i.e. `data/raw/eth`). The train/validation/test splits are the same as those found in [Social GAN](https://github.com/agrimgupta92/sgan).

If you want the *original* ETH or UCY datasets, you can find them here: [ETH Dataset](http://www.vision.ee.ethz.ch/en/datasets/) and [UCY Dataset](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data).
