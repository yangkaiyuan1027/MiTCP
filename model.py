from typing import List, Union, Tuple

import numpy as np
import pandas as pd
import json
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
from chemprop.nn_utils import get_activation_function, initialize_weights


class MoleculeModel(nn.Module):
    """A :class:`MoleculeModel` is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, args: TrainArgs):
        """
        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        super(MoleculeModel, self).__init__()

        self.classification = args.dataset_type == 'classification'
        self.multiclass = args.dataset_type == 'multiclass'
        self.loss_function = args.loss_function

        if hasattr(args, 'train_class_sizes'):
            self.train_class_sizes = args.train_class_sizes
        else:
            self.train_class_sizes = None

        # when using cross entropy losses, no sigmoid or softmax during training. But they are needed for mcc loss.
        if self.classification or self.multiclass:
            self.no_training_normalization = args.loss_function in ['cross_entropy', 'binary_cross_entropy']

        self.output_size = args.num_tasks
        if self.multiclass:
            self.output_size *= args.multiclass_num_classes
        if self.loss_function == 'mve':
            self.output_size *= 2  # return means and variances
        if self.loss_function == 'dirichlet' and self.classification:
            self.output_size *= 2  # return dirichlet parameters for positive and negative class
        if self.loss_function == 'evidential':
            self.output_size *= 4  # return four evidential parameters: gamma, lambda, alpha, beta

        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)

        if self.loss_function in ['mve', 'evidential', 'dirichlet']:

            self.softplus = nn.Softplus()

        self.create_encoder(args)
        self.create_ffn(args)

        self.gene2id = None
        self.gene_ctp_list = None
        self.gene_ctp_ids = None

        self.GCN1 = GCNConv(300, 300)
        self.GCN2 = GCNConv(300, 300)
        self.edge_index = None
        self.extra_data_readin(args)
        self.gene_embedding_layer = nn.Embedding(len(self.gene_ctp_list),300)
        # self.gene_nets = nn.ModuleList([nn.Linear(100,1) for i in range(len(self.gene_ctp_list))])
        # self.mol_ffn = nn.Linear(2348,300)
        self.ffn1 = nn.Linear(600,512)
        self.ffn2 = nn.Linear(512,512)
        self.ffn3 = nn.Linear(512,512)
        self.ffn4 = nn.Linear(512, 978)
        self.dropout = nn.Dropout(0.2)

        initialize_weights(self)
        self.device = args.device

    def extra_data_readin(self, args: TrainArgs) -> None:
        train = pd.read_csv(args.save_dir + '/train_full.csv', index_col=0)
        self.gene_ctp_list = list(train.columns) #
        self.gene2id = {gene: i for i, gene in enumerate(self.gene_ctp_list)}

        #construct gene coexpression network based on the training data
        correlation_matrix = train.corr(method='pearson')
        corr_threshold = 0.4
        edges = []
        # edge_weight = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > corr_threshold:
                    edges.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
                    # edge_weight.append(corr)
        edges = np.array(edges)

        self.gene2id = {gene: i for i, gene in enumerate(self.gene_ctp_list)}
        self.gene_ctp_ids = torch.LongTensor([self.gene2id[gene] for gene in self.gene_ctp_list]).to(args.device)

        edge_index = np.vectorize(self.gene2id.__getitem__)(edges)
        edge_index = torch.from_numpy(edge_index.T)
        edge_index = edge_index.to(torch.long).contiguous()
        self.edge_index = edge_index.to(args.device)

    def create_encoder(self, args: TrainArgs) -> None:
        """
        Creates the message passing encoder for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.encoder = MPN(args)

        if args.checkpoint_frzn is not None:
            if args.freeze_first_only:  # Freeze only the first encoder
                for param in list(self.encoder.encoder.children())[0].parameters():
                    param.requires_grad = False
            else:  # Freeze all encoders
                for param in self.encoder.parameters():
                    param.requires_grad = False


    def create_ffn(self, args: TrainArgs) -> None:
        """
        Creates the feed-forward layers for the model.

        :param args: A :class:`~chemprop.args.TrainArgs` object containing model arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            if args.reaction_solvent:
                first_linear_dim = args.hidden_size + args.hidden_size_solvent
            else:
                first_linear_dim = args.hidden_size * args.number_of_molecules
            if args.use_input_features:
                first_linear_dim += args.features_size

        if args.atom_descriptors == 'descriptor':
            first_linear_dim += args.atom_descriptors_size

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, self.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, self.output_size),
            ])

        # If spectra model, also include spectra activation
        if args.dataset_type == 'spectra':
            if args.spectra_activation == 'softplus':
                spectra_activation = nn.Softplus()
            else:  # default exponential activation which must be made into a custom nn module
                class nn_exp(torch.nn.Module):
                    def __init__(self):
                        super(nn_exp, self).__init__()

                    def forward(self, x):
                        return torch.exp(x)

                spectra_activation = nn_exp()
            ffn.append(spectra_activation)

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

        if args.checkpoint_frzn is not None:
            if args.frzn_ffn_layers > 0:
                for param in list(self.ffn.parameters())[0:2 * args.frzn_ffn_layers]:  # Freeze weights and bias for given number of layers
                    param.requires_grad = False


    def fingerprint(self,
                    batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                    features_batch: List[np.ndarray] = None,
                    atom_descriptors_batch: List[np.ndarray] = None,
                    atom_features_batch: List[np.ndarray] = None,
                    bond_features_batch: List[np.ndarray] = None,
                    fingerprint_type: str = 'MPN') -> torch.Tensor:
        """
        Encodes the latent representations of the input molecules from intermediate stages of the model.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param fingerprint_type: The choice of which type of latent representation to return as the molecular fingerprint. Currently
                                 supported MPN for the output of the MPNN portion of the model or last_FFN for the input to the final readout layer.
        :return: The latent fingerprint vectors.
        """
        if fingerprint_type == 'MPN':
            return self.encoder(batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch)
        elif fingerprint_type == 'last_FFN':
            return self.ffn[:-1](self.encoder(batch, features_batch, atom_descriptors_batch,
                                              atom_features_batch, bond_features_batch))
        else:
            raise ValueError(f'Unsupported fingerprint type {fingerprint_type}.')


    def forward(self,
                batch: Union[List[List[str]], List[List[Chem.Mol]], List[List[Tuple[Chem.Mol, Chem.Mol]]], List[BatchMolGraph]],
                features_batch: List[np.ndarray] = None,
                atom_descriptors_batch: List[np.ndarray] = None,
                atom_features_batch: List[np.ndarray] = None,
                bond_features_batch: List[np.ndarray] = None) -> torch.FloatTensor:
        """
        Runs the :class:`MoleculeModel` on input.

        :param batch: A list of list of SMILES, a list of list of RDKit molecules, or a
                      list of :class:`~chemprop.features.featurization.BatchMolGraph`.
                      The outer list or BatchMolGraph is of length :code:`num_molecules` (number of datapoints in batch),
                      the inner list is of length :code:`number_of_molecules` (number of molecules per datapoint).
        :param features_batch: A list of numpy arrays containing additional features.
        :param atom_descriptors_batch: A list of numpy arrays containing additional atom descriptors.
        :param atom_features_batch: A list of numpy arrays containing additional atom features.
        :param bond_features_batch: A list of numpy arrays containing additional bond features.
        :return: The output of the :class:`MoleculeModel`, containing a list of property predictions
        """

        # code of chemprop
        # output = self.ffn(self.encoder(batch, features_batch, atom_descriptors_batch,
        #                                atom_features_batch, bond_features_batch))

        # gene embedding
        output = self.encoder(batch, features_batch, atom_descriptors_batch,
                                       atom_features_batch, bond_features_batch) #molecule embedding
        # output = self.mol_ffn(output)
        # output = torch.relu(output)
        batch_size = output.shape[0]

        gene_embeddings = self.gene_embedding_layer(self.gene_ctp_ids)
        gene_embeddings = self.GCN1(gene_embeddings, edge_index=self.edge_index)
        gene_embeddings = self.GCN2(gene_embeddings, edge_index=self.edge_index)
        gene_embeddings = gene_embeddings.unsqueeze(0)
        gene_embeddings = gene_embeddings.repeat(batch_size,1,1)

        output = output.unsqueeze(1)
        output = output.repeat(1,len(self.gene_ctp_list),1)

        gene_mol = torch.cat([gene_embeddings, output], dim=2)
        gene_mol = self.ffn1(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn2(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn3(gene_mol)
        gene_mol = torch.relu(gene_mol)
        gene_mol = self.dropout(gene_mol)
        gene_mol = self.ffn4(gene_mol)
        output = torch.diagonal(gene_mol, dim1=-2, dim2=-1)

        if self.classification and not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.shape[0], -1, self.num_classes))  # batch size x num targets x num classes per target
            if not (self.training and self.no_training_normalization) and self.loss_function != 'dirichlet':
                output = self.multiclass_softmax(output)  # to get probabilities during evaluation, but not during training when using CrossEntropyLoss

        # Modify multi-input loss functions
        if self.loss_function == 'mve':
            means, variances = torch.split(output, output.shape[1] // 2, dim=1)
            variances = self.softplus(variances)
            output = torch.cat([means, variances], axis=1)
        if self.loss_function == 'evidential':
            means, lambdas, alphas, betas = torch.split(output, output.shape[1]//4, dim=1)
            lambdas = self.softplus(lambdas)  # + min_val
            alphas = self.softplus(alphas) + 1  # + min_val # add 1 for numerical contraints of Gamma function
            betas = self.softplus(betas)  # + min_val
            output = torch.cat([means, lambdas, alphas, betas], dim=1)
        if self.loss_function == 'dirichlet':
            output = nn.functional.softplus(output) + 1

        return output


class GeneRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

class GeneRegressorList(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_genes):
        super(GeneRegressorList, self).__init__()
        self.gene_nets = nn.ModuleList([GeneRegressor(input_dim, hidden_dim, output_dim) for i in range(num_genes)])

    def forward(self, x):
        output = torch.empty([0,len(self.gene_nets)]).cuda()
        for batch in range(x.shape[0]):
            gene_preds = []
            for i, gene_net in enumerate(self.gene_nets):
                gene_pred = gene_net(x[batch][i])
                gene_preds.append(gene_pred)
            gene_preds = torch.stack(gene_preds, dim=1)
            output = torch.cat([output,gene_preds],dim = 0)
        return output