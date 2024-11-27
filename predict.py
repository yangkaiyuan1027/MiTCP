from typing import List

import torch
from tqdm import tqdm
import numpy as np

from args import PredictArgs, TrainArgs
from model import MoleculeModel
from make_predictions import make_predictions

arguments = [
    '--checkpoint_dir', 'coexpression/GCN_0.4/',
   '--test_path', 'IPF_drugs.csv',
  '--preds_path', 'IPF_preds.csv',
]

args = PredictArgs().parse_args(arguments)
make_predictions(args=args)
