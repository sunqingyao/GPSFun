import numpy as np
import pandas as pd
from tqdm import tqdm
import os, argparse, datetime, traceback

import torch
import torch_geometric
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from feature_extraction.ProtTrans import get_ProtTrans
from feature_extraction.process_structure import get_pdb_xyz, process_dssp, match_dssp
from utils import *
from model_GPSite import *
from model_SPROF_GO import *
from model_GPSLoc import *
from model_GPSol import *


############ Set to your own path! ############
ProtTrans_path = "/app/common/bio-platform/GPSFun/Prot-T5-XL-U50/"
###############################################

script_path = os.path.split(os.path.realpath(__file__))[0] + "/"
model_path = os.path.dirname(script_path[0:-1]) + "/model/"


def extract_feat(ID_list, seq_list, outpath, gpu):
    #max_len = max([len(seq) for seq in seq_list])
    #chunk_size = 32 if max_len > 1000 else 64
    chunk_size = 32
    max_token = 800

    esmfold_cmd = "python {}/feature_extraction/esmfold.py -i {} -o {} --chunk-size {} --max-tokens-per-batch {} --cpu-offload".format(script_path, outpath + "test_seq_short.fa", outpath + "pdb/", chunk_size, max_token)
    if not gpu: # slow!!
        esmfold_cmd += " --cpu-only"
    else:
        esmfold_cmd = "CUDA_VISIBLE_DEVICES=" + gpu + " " + esmfold_cmd
    os.system(esmfold_cmd + " | tee {}/esmfold_pred.log".format(outpath))

    success_ID = [file[0:-4] for file in os.listdir(outpath + "pdb/")] # 以防有接近但小于MAX_SEQ_LEN_ESMFold的蛋白失败了
    success_ID = [ID for ID in ID_list if ID in success_ID] # 保留原来的顺序

    get_ProtTrans(ID_list, seq_list, ProtTrans_path, outpath, gpu)

    print("Processing PDB files...")
    for ID in tqdm(success_ID):
        with open(outpath + "pdb/" + ID + ".pdb", "r") as f:
            X = get_pdb_xyz(f.readlines()) # [L, 5, 3]
        torch.save(torch.tensor(X, dtype = torch.float32), outpath + "pdb/" + ID + '.tensor')

    print("Extracting DSSP features...")
    for i in tqdm(range(len(ID_list))):
        ID = ID_list[i]
        seq = seq_list[i][0:MAX_SEQ_LEN_ESMFold]
        if ID not in success_ID:
            continue

        os.system("{}/feature_extraction/mkdssp -i {}/pdb/{}.pdb -o {}/DSSP/{}.dssp".format(script_path, outpath, ID, outpath, ID))
        dssp_seq, dssp_matrix, dssp_feature_text = process_dssp("{}/DSSP/{}.dssp".format(outpath, ID))
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)

        torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "{}/DSSP/{}.tensor".format(outpath, ID))
        with open("{}/DSSP/{}_SS.txt".format(outpath, ID), "w") as f:
            f.write(dssp_feature_text)
        os.system("rm {}/DSSP/{}.dssp".format(outpath, ID))

    return success_ID


def GPSite_predict(ID_list, outpath, batch_size, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    node_input_dim = GPSite_config['node_input_dim']
    edge_input_dim = GPSite_config['edge_input_dim']
    hidden_dim = GPSite_config['hidden_dim']
    layer = GPSite_config['layer']
    augment_eps = GPSite_config['augment_eps']
    dropout = GPSite_config['dropout']

    task_list = ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]

    # Test
    test_dataset = GPSite_Dataset(ID_list, outpath)
    test_dataloader = pyg_DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load(model_path + 'GPSite/fold%s.ckpt'%fold, device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, layer, augment_eps, dropout, task_list).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {}
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            outputs = [model(data.X, data.node_feat, data.edge_index, data.batch).sigmoid() for model in models]
            outputs = torch.stack(outputs,0).mean(0) # average the predictions from 5 models

        IDs = data.name
        outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = []
            for j in range(len(task_list)):
                test_pred_dict[ID].append(list(outputs_split[i][:,j].detach().cpu().numpy()))

    return test_pred_dict


def SPROF_GO_predict(ID_list, outpath, batch_size, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    test_dataset = SPROF_GO_Dataset(ID_list, outpath)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, collate_fn = test_dataset.collate_fn, shuffle = False, drop_last = False, num_workers = 4)

    predictions = []
    for task in ["MF", "BP", "CC"]:
        print("predicting {}...".format(task))

        # Load SPROF-GO models
        models = []
        for i in range(5):
            model = SPROF_GO(task, SPROF_GO_config["feature_dim"], SPROF_GO_config["hidden_dim"], SPROF_GO_config["num_emb_layers"], SPROF_GO_config["num_heads"], SPROF_GO_config["dropout"], device).to(device)
            state_dict = torch.load(model_path + 'SPROF-GO/{}/{}_model_{}.ckpt'.format(task, task, i), device)
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)

        # Make predictions
        test_pred = []
        for batch in tqdm(test_dataloader):
            protein_feat, protein_mask = [x.to(device) for x in batch]
            with torch.no_grad():
                preds = [model(protein_feat, protein_mask) for model in models]
                preds = torch.stack(preds, 0).mean(0) # average predictions from 5 models
                test_pred.append(preds.detach().cpu().numpy())
        test_pred = np.concatenate(test_pred) # initial prediction

        # Label diffusion
        test_fasta = outpath + "test_seq.fa"
        diffusion_pred = LabelDiffusion(task, test_pred, test_fasta, outpath) # final prediction
        predictions.append(np.round(diffusion_pred, decimals = 3))

    return predictions


def GPSLoc_predict(ID_list, outpath, batch_size, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    node_input_dim = GPSLoc_config['node_input_dim']
    edge_input_dim = GPSLoc_config['edge_input_dim']
    hidden_dim = GPSLoc_config['hidden_dim']
    layer = GPSLoc_config['layer']
    dropout = GPSLoc_config['dropout']

    df_output = pd.DataFrame(columns = ["ID", "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane", "Mitochondrion",
                            "Plastid", "Endoplasmic reticulum", "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome"])
    test_dataset = GPSLoc_Dataset(ID_list, outpath)
    test_dataloader = pyg_DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load(model_path + 'GPSLoc/fold%s.ckpt'%fold, device)
        model = GPSLoc(node_input_dim, edge_input_dim, hidden_dim, layer, dropout).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {}
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            outputs = [model(data).sigmoid() for model in models]
            avg_outputs = list(np.round(torch.stack(outputs,0).mean(0).detach().cpu().numpy(), decimals = 3))

        IDs = data.name
        for i, ID in enumerate(IDs):
            df_output.loc[len(df_output.index)] = [ID] + list(avg_outputs[i])
    df_output.to_csv(outpath + "/pred_GPSLoc/SL_preds.csv", index=False)


def GPSol_predict(ID_list, outpath, batch_size, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    node_input_dim = GPSol_config['node_input_dim']
    edge_input_dim = GPSol_config['edge_input_dim']
    hidden_dim = GPSol_config['hidden_dim']
    layer = GPSol_config['layer']
    dropout = GPSol_config['dropout']

    test_dataset = GPSol_Dataset(ID_list, outpath)
    test_dataloader = pyg_DataLoader(test_dataset, batch_size = batch_size, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load(model_path + 'GPSol/fold%s.ckpt'%fold, device)
        model = GPSol(node_input_dim, edge_input_dim, hidden_dim, layer, dropout, device).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    outfile = "ID,Solubility\n"
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            outputs = [model(data.X, data.node_feat, data.edge_index, data.batch).sigmoid() for model in models]
            outputs = torch.stack(outputs,0).mean(0).detach().cpu().numpy()

        IDs = data.name
        for i, ID in enumerate(IDs):
            outfile += '{},{:.3f}\n'.format(ID, outputs[i])

    with open(outpath + "/pred_GPSol/Sol_preds.csv", "w") as f:
        f.write(outfile)


def main(seq_info, outpath, batch, gpu):
    ID_list, seq_list = seq_info
    id2seq = dict(zip(ID_list, seq_list))
    for dir_name in ["pdb", "ProtTrans", "DSSP", "pred_GPSite", "pred_SPROF-GO", "pred_GPSLoc", "pred_GPSol"]:
        os.makedirs(outpath + dir_name, exist_ok = True)

    print("\n######## Feature extraction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    success_ID = extract_feat(ID_list, seq_list, outpath, gpu)

    print("\n######## Feature extraction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))


    ######### GPSite #########
    print("\n######## GPSite prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    predictions = GPSite_predict(success_ID, outpath, batch, gpu)

    print("\n######## GPSite prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    export_GPSite_predictions(predictions, id2seq, outpath)

    print("\n######## GPSite results are saved in {} ########\n".format(outpath + "pred_GPSite/"))


    ######### SPROF-GO #########
    print("\n######## SPROF-GO prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    predictions = SPROF_GO_predict(ID_list, outpath, batch, gpu)

    print("\n######## SPROF-GO prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    export_SPROF_GO_predictions(predictions, ID_list, success_ID, outpath, script_path)

    print("\n######## SPROF-GO results are saved in {} ########\n".format(outpath + "pred_SPROF-GO/GO_preds.txt"))


    ######### GPSLoc #########
    print("\n######## GPSLoc prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    GPSLoc_predict(success_ID, outpath, batch, gpu)

    print("\n######## GPSLoc prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
    print("\n######## GPSLoc results are saved in {} ########\n".format(outpath + "pred_GPSLoc/SL_preds.csv"))

    ######### GPSol #########
    print("\n######## GPSol prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    GPSol_predict(success_ID, outpath, batch, gpu)

    print("\n######## GPSol prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
    print("\n######## GPSol results are saved in {} ########\n".format(outpath + "pred_GPSol/Sol_preds.csv"))

    get_similar_proteins(script_path, outpath)

    # MakeFileForDownload(outpath)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("-f", "--fasta_file", type = str, default = None, help = "Input FASTA file")
        parser.add_argument("-s", "--fasta_str", type = str, default = None, help = "Input FASTA string")
        parser.add_argument("-o", "--outpath", type = str, help = "Output path to save intermediate files and final predictions", required=True)
        parser.add_argument("-b", "--batch", type = int, default = 4, help = "Batch size for GPSFun prediction")
        parser.add_argument("--gpu", type = str, default = None, help = "The GPU id used for feature extraction and GPSFun prediction")

        args = parser.parse_args()
        outpath = args.outpath.rstrip("/") + "/"
        os.makedirs(outpath, exist_ok = True)

        if args.fasta_file:
            seq_info = process_fasta(args.fasta_file, outpath, is_file = True)
        elif args.fasta_str:
            seq_info = process_fasta(args.fasta_str, outpath, is_file = False)
        else:
            seq_info = None

        if seq_info == None:
            print("Please paste the FASTA sequences in the text box or upload a FASTA file!")
        elif seq_info == -1:
            print("The format of your input FASTA file is incorrect! Please check!")
        elif seq_info == 1:
            print("Too much sequences! Up to {} sequences are supported each time!".format(MAX_INPUT_SEQ))
        else:
            main(seq_info, outpath, args.batch, args.gpu)
    except:
        print(traceback.format_exc())
        print("Please check if your input format is the same as our example, or contact the authors of GPSFun.")
