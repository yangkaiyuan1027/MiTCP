import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import chemprop
from cross_validate import cross_validate
from run_training import run_training

arguments = [
    '--data_path', 'ctp_phase1+2.csv',
    '--dataset_type', 'regression',
    '--save_dir', 'checkpoints'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
args.save_smiles_splits = True
args.save_preds = True
args.epochs = 400
args.batch_size = 256
args.dropout = 0.2
args.ffn_hidden_size = 512
args.ffn_num_layers = 4
# args.num_folds = 5
args.loss_function = 'mse_dir'
# args.features_generator = ['morgan_count']
mean_score, std_score = cross_validate(args=args, train_func=run_training)