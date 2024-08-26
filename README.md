# MiTCP
The code implementation of 'Predicting transcriptional changes induced by molecules with MiTCP' 

## Dependencies
Our code environment is python 3.7 on Ubuntu 20.04 with CUDA 11.5. The packages of our environment which are dependencies for running DGP-AMIO are provided as follows:
* numpy==1.21.6
* pandas==1.15
* scikit-learn==1.0.2
* tqdm==4.64.1
* chemprop==1.5.2
* torch==1.10.2+cu113
* torch_geometric==2.1.0
* rdkit==2022.9.1

All of the packages can be installed through pip. Although not necessary, we strongly recommend GPU acceleration and conda for package management.

## Run MiTCP
If you want to run MiTCP, simply run:
```
python train_MiTCP.py
```
