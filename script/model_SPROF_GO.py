import os, pickle, time
import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as ssp
from scipy.sparse.linalg import inv

utility_path = os.path.split(os.path.realpath(__file__))[0] + "/utility_files/SPROF-GO/"
dataset_path = utility_path + "label_diffusion/"

##############  Hyperparameter  ##############
SPROF_GO_config = {
    'feature_dim': 1024,
    'hidden_dim': 256,
    'num_emb_layers': 2,
    'num_heads': 8,
    'dropout': 0.1
}


##############  DataLoader  ##############
class SPROF_GO_Dataset:
    def __init__(self, ID_list, outpath):
        self.IDs = ID_list
        self.outpath = outpath
        self.feat_dim = SPROF_GO_config['feature_dim']
        self.Min_protrans = torch.tensor(np.load(utility_path + "Min_ProtTrans_repr.npy"), dtype = torch.float32)
        self.Max_protrans = torch.tensor(np.load(utility_path + "Max_ProtTrans_repr.npy"), dtype = torch.float32)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, idx):
        ID = self.IDs[idx]
        protein_feat = torch.load(self.outpath + "ProtTrans/" + ID + ".tensor")
        protein_feat = (protein_feat - self.Min_protrans) / (self.Max_protrans - self.Min_protrans) # normalization
        return protein_feat

    def padding(self, batch):
        maxlen = max([protein_feat.shape[0] for protein_feat in batch])
        batch_protein_feat = []
        batch_protein_mask = []
        for protein_feat in batch:
            padded_protein_feat = torch.zeros(maxlen, self.feat_dim)
            padded_protein_feat[:protein_feat.shape[0]] = protein_feat
            batch_protein_feat.append(padded_protein_feat)

            protein_mask = torch.zeros(maxlen)
            protein_mask[:protein_feat.shape[0]] = 1
            batch_protein_mask.append(protein_mask)

        return torch.stack(batch_protein_feat), torch.stack(batch_protein_mask)

    def collate_fn(self, batch):
        protein_feat, protein_mask = self.padding(batch) # [B, maxlen, hid], [B, maxlen]
        return protein_feat, protein_mask


##############  Model  ##############
class SPROF_GO(nn.Module):
    def __init__(self, task, feature_dim=1024, hidden_dim=256, num_emb_layers=2, num_heads=8, dropout=0.1, device = torch.device('cpu')):
        super(SPROF_GO, self).__init__()

        # Child Matrix: CM_ij = 1 if the jth GO term is a subclass of the ith GO term
        self.CM = torch.tensor(ssp.load_npz(utility_path + task + "_CM.npz").toarray()).to(device)

        # Embedding layers
        self.input_block = nn.Sequential(
                                         nn.LayerNorm(feature_dim, eps=1e-6)
                                        ,nn.Linear(feature_dim, hidden_dim)
                                        ,nn.LeakyReLU()
                                        )

        self.hidden_block = []
        for i in range(num_emb_layers - 1):
            self.hidden_block.extend([
                                      nn.LayerNorm(hidden_dim, eps=1e-6)
                                     ,nn.Dropout(dropout)
                                     ,nn.Linear(hidden_dim, hidden_dim)
                                     ,nn.LeakyReLU()
                                     ])
            if i == num_emb_layers - 2:
                self.hidden_block.extend([nn.LayerNorm(hidden_dim, eps=1e-6)])

        self.hidden_block = nn.Sequential(*self.hidden_block)

        # Self-attention pooling layer
        self.ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, num_heads)
                                 )

        # Output layer
        self.label_size = {"MF":790, "BP":4766, "CC":667}[task] # terms with >= 50 samples in the training + validation sets
        self.output_block = nn.Sequential(
                                         nn.Linear(num_heads*hidden_dim, num_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(num_heads*hidden_dim, eps=1e-6)
                                         ,nn.Dropout(dropout)
                                         ,nn.Linear(num_heads*hidden_dim, self.label_size)
                                         )

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, h_V, mask, y = None):
        h_V = self.input_block(h_V)
        h_V = self.hidden_block(h_V)

        # Multi-head self-attention pooling
        att = self.ATFC(h_V)    # [B, L, num_heads]
        att = att.masked_fill(mask[:, :, None] == 0, -1e9)
        att = F.softmax(att, dim=1)
        att = att.transpose(1,2)   # [B, num_heads, L]
        h_V = att@h_V    # [B, num_heads, hidden_dim]
        h_V = torch.flatten(h_V, start_dim=1) # [B, num_heads*hidden_dim]

        h_V = self.output_block(h_V).sigmoid() # [B, label_size]

        # Hierarchical learning
        if self.training:
            a = (1 - y) * torch.max(h_V.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0] # the probability of a negative class should take the maximal probabilities of its subclasses
            b = y * torch.max(h_V.unsqueeze(1) * (self.CM.unsqueeze(0) * y.unsqueeze(1)), dim = -1)[0] # the probability of a positive class should take the maximal probabilities of its positive subclasses
            h_V = a + b
        else:
            h_V = torch.max(h_V.unsqueeze(1) * self.CM.unsqueeze(0), dim = -1)[0]  # [B, 1, label_size] * [1, label_size, label_size]

        return h_V.float() # [B, label_size]


##############  Label diffusion  ##############
def sparse_divide_nonzero(a, b):
    inv_b = b.copy()
    inv_b.data = 1 / inv_b.data
    return a.multiply(inv_b)


def jaccard(Q):
    Co = Q.dot(Q.T)
    CLEN = Q.sum(axis=1) # numpy matrix, shape = (N, 1)

    # original implementation: J = Co / (CLEN + CLEN.T - Co + 1)
    nonzero_mask = Co.astype('bool')
    denominator = nonzero_mask.multiply(CLEN) + nonzero_mask.multiply(CLEN.T) - Co + nonzero_mask

    J = sparse_divide_nonzero(Co, denominator)
    return J


def compute_L(Q):
    JQ = jaccard(Q).multiply(Q)
    degree = 1.0 / JQ.sum(axis=1) # shape = (N, 1)
    Q_star = 0.5 * (JQ.multiply(degree) + JQ.multiply(degree.T))

    degree = Q_star.sum(axis=1)
    D = ssp.spdiags(degree.T, 0, Q.shape[0], Q.shape[0])
    L = D - Q_star

    return L


def homology_matrix(fasta, cutoff = 0.1):
    ID = []
    with open(fasta, "r") as f:
        lines = f.readlines()
    for line in lines:
        if line[0] == ">":
            ID.append(line[1:-1])

    ID2idx = dict(zip(ID, range(len(ID))))

    os.system("{}diamond makedb --in {} -d {} --quiet".format(utility_path, fasta, fasta))
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {} --very-sensitive --quiet -p 8'.format(utility_path, fasta, fasta, fasta + ".tsv"))

    with open(fasta + ".tsv", "r") as f:
        out = f.readlines()

    os.system("rm {}.*".format(fasta))

    homology = {}
    for line in out:
        fields = line.strip().split()
        query_idx = ID2idx[fields[0]]
        subject_idx = ID2idx[fields[1]]
        identity = float(fields[2]) / 100

        if query_idx not in homology:
            homology[query_idx] = {}
        if subject_idx in homology[query_idx]:
            homology[query_idx][subject_idx] = max(homology[query_idx][subject_idx], identity)
        else:
            homology[query_idx][subject_idx] = identity

        # handle symmetry
        if subject_idx not in homology:
            homology[subject_idx] = {}
        if query_idx in homology[subject_idx]:
            homology[subject_idx][query_idx] = max(homology[subject_idx][query_idx], identity)
        else:
            homology[subject_idx][query_idx] = identity

    row = []
    col = []
    data = []
    for i in homology:
        for j in homology[i]:
            val = homology[i][j]

            if i == j:
                val = 1.0
            if val < cutoff:
                continue

            row.append(i)
            col.append(j)
            data.append(val)

    data = np.array(data)
    graph = ssp.csr_matrix((data, (row, col)), shape = (len(ID), len(ID)))

    return graph


def LabelDiffusion(task, initial_pred, test_fasta, outpath, lamda = 1, identity_cutoff = 0.1):
    '''
    This is an efficient implementation of the label diffusion algorithm
    by S2F, using DIAMOND and the sparse matrix throughout the caculation.
    '''
    print("Start diffusion...")
    start = time.time()

    # prepare the training set
    with open(dataset_path + task + "_train.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(dataset_path + task + "_valid.pkl", "rb") as f:
        valid_dataset = pickle.load(f)
    train_dataset = {**train_dataset, **valid_dataset}

    train_fasta = dataset_path + task + "_train+valid.fa"
    if not os.path.exists(train_fasta):
        fasta_file = ""
        for ID in train_dataset:
            fasta_file += (">" + ID + "\n" + train_dataset[ID][0] + "\n")
        with open(train_fasta, "w") as f:
            f.write(fasta_file)


    # search for hits/seeds in the training set for the test proteins
    os.system("{}diamond makedb --in {} -d {} --quiet".format(utility_path, train_fasta, outpath + task))
    os.system('{}diamond blastp -d {}.dmnd -q {} -o {}testVStrain.tsv --very-sensitive --quiet -p 8'.format(utility_path, outpath + task, test_fasta, outpath))
    os.system("rm {}.dmnd".format(outpath + task))

    with open(outpath + "testVStrain.tsv", "r") as f:
        testVStrain = f.readlines()

    train_seed_ID = set()
    for line in testVStrain:
        fields = line.strip().split()
        subject_id = fields[1]
        identity = float(fields[2]) / 100
        if identity > identity_cutoff:
            train_seed_ID.add(subject_id)

    os.system("rm {}testVStrain.tsv".format(outpath))

    if len(initial_pred) == 1 and len(train_seed_ID) == 0: # only one test seq and no train seed found
        end = time.time()
        print("Diffusion done! Cost {}s.".format(end - start))
        return initial_pred

    train_seed_ID_list = []
    train_seed_fasta = ""
    for ID in train_dataset: # keep the original ID order
        if ID in train_seed_ID:
            train_seed_ID_list.append(ID)
            train_seed_fasta += (">" + ID + "\n" + train_dataset[ID][0] + "\n")

    with open(outpath + "train_seed_seq.fa", "w") as f:
        f.write(train_seed_fasta)


    # get the homology matrix Q for the test proteins + seeds in the training set
    os.system("cat {}train_seed_seq.fa {} > {}train_seed_and_test.fa".format(outpath, test_fasta, outpath))
    os.system("rm {}train_seed_seq.fa".format(outpath))

    Q = homology_matrix(outpath + "train_seed_and_test.fa")
    os.system("rm {}train_seed_and_test.fa".format(outpath))


    # get labels of train seeds
    row = []
    col = []
    for i, ID in enumerate(train_seed_ID_list):
        label_idx = train_dataset[ID][1]
        row += [i] * len(label_idx)
        col += label_idx
    data = [1] * len(row)

    label_size = {"MF":790, "BP":4766, "CC":667}[task]
    train_seed_label = ssp.csr_matrix((data, (row, col)), shape = (len(train_seed_ID_list), label_size))


    # Label Diffusion
    initial_pred = ssp.csr_matrix(initial_pred)
    initial_pred = ssp.vstack([train_seed_label, initial_pred])

    L = compute_L(Q)
    IlambdaL = ssp.identity(Q.shape[0]) + L.multiply(lamda)
    kernel = inv(IlambdaL.tocsc())[len(train_seed_ID_list):]
    diffusion_pred = kernel.dot(initial_pred).toarray()

    end = time.time()
    print("Diffusion done! Cost {}s.".format(end - start))

    return diffusion_pred
