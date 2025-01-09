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

## Train MiTCP
If you want to train MiTCP, simply run:
```
python train_MiTCP.py
```

## Load trained model and predict CTPs of molecules
If you want to predict CTPs (changes of transcriptional profiles) of your own molecules, run:
```
python predict.py
```
This command will load the trained model parameters and predict the CTPs of the molecules you provide. We have uploaded our trained model parameter file 'model.pt', and we also provide the training data the 'model.pt' was trained on (in the predicting process the training data need to be provided because the model needs to construct the coexpression graph based on the training data). All you have to do is to prepare SMILES of molecules in a CSV file and specify the path to save the predictions in predict.py.
The CSV file containing molecular SMILES is required to include a column named 'SMILES'. We also provide a small-scale example file, mol_SMILES.csv, for users to run and familiarize themselves with MiTCP's prediction.
## Preproccessed L1000 data
The preproccessed L1000 data can be downloaded at https://zenodo.org/records/13991478?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImIwOTNjODFmLTg3YTktNGU4ZC05MGVjLWZjOTE3NzNhYjA2ZSIsImRhdGEiOnt9LCJyYW5kb20iOiIyMDBiODY4OTNmOGE3Y2I0YzFhNGEyNDMzZDYxOTM1ZSJ9.UEeNjYhmIhWnjf5mxhxzjKyYWg2hZr8sIlRFLo86w6C_kSLZALgKnTj7oKqkfW8iHH1tN4E5SyvL2kbGE-zMdw
