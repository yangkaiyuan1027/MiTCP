from typing import List

import torch
from tqdm import tqdm
import numpy as np

from args import PredictArgs, TrainArgs
from model import MoleculeModel
from make_predictions import make_predictions

arguments = [
    '--checkpoint_path', 'model.pt',
   '--test_path', 'mol_SMILES.csv',
  '--preds_path', 'CTP_preds.csv',
]

args = PredictArgs().parse_args(arguments)
make_predictions(args=args)
