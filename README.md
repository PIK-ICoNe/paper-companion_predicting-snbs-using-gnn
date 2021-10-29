# predicting-snbs-using-gnn_paper
Predicting Dynamic Stability of Power Grids using Graph Neural Networks

This repository contains the source-code of the paper, but does not contain any data. The datasets including all scripts to reproduce the results in the paper are given at https://zenodo.org/record/5148085. In this repository, there are scripts in Julia to generate the dataset and prepare the data for PyTorch and also the python-scripts to train the ML-models. 


# Bugs in code

There are limitations of the code, please let me know if you are interested in using the code, then I would fix them. The known limitations are:
- Cuda is not supported
- Changing batch sizes leads to wrong results
