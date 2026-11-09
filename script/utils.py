import string, os, pickle
import numpy as np

MAX_INPUT_SEQ = 20
MAX_SEQ_LEN_ESMFold = 850 # Currently, seq > 850 aa will triger OOM error
MAX_SEQ_LEN_SPROF_GO = 2000


# deal with IDs with different formats: e.g. "sp|P05067|A4_HUMAN Amyloid-beta precursor protein" (UniProt), "7PRW_1|Chains A, B|Glucocorticoid receptor|Homo sapiens" (PDB)
def get_ID(name):
    name = name.split("|")
    ID = "_".join(name[0:min(2, len(name))])
    ID = ID.replace(" ", "_")
    return ID


def remove_non_standard_aa(seq):
    standard_aa = "ACDEFGHIKLMNPQRSTVWY"
    new_seq = ""
    for aa in seq:
        if aa in standard_aa:
            new_seq += aa
    return new_seq


def process_fasta(fasta, outpath, is_file):
    ID_list = []
    seq_list = []

    if is_file:
        with open(fasta, "r") as f:
            records = f.read().split(">")[1:]
    else:
        records = fasta.split(">")[1:]

    for item in records:
        lines = item.strip().split("\n")
        ID_list.append(get_ID(lines[0]))
        seq = "".join(lines[1:])
        seq = remove_non_standard_aa(seq.upper())
        seq_list.append(seq[0:min(MAX_SEQ_LEN_SPROF_GO, len(seq))]) # trim long sequence # 先剪短到SPROF-GO的长度，跑ProtTrans，后续再剪到ESMFold的长度

    if len(ID_list) == len(seq_list):
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            new_fasta = "" # with processed IDs and seqs # 用来跑label diffusion
            new_fasta_short = "" # 用来跑ESMFold
            for i in range(len(ID_list)):
                new_fasta += (">" + ID_list[i] + "\n" + seq_list[i] + "\n")
                new_fasta_short += (">" + ID_list[i] + "\n" + seq_list[i][0:MAX_SEQ_LEN_ESMFold] + "\n")
            with open(outpath + "test_seq.fa", "w") as f:
                f.write(new_fasta)
            with open(outpath + "test_seq_short.fa", "w") as f:
                f.write(new_fasta_short)

            return [ID_list, seq_list]
    else:
        return -1


def get_similar_proteins(script_path, outpath):
    os.system("{}diamond blastp -d {}.dmnd -q {} -o {}similar_proteins_raw.tsv --very-sensitive --quiet -p 8".format(script_path + "utility_files/SPROF-GO/", script_path + "utility_files/similar_proteins/sprot", outpath + "test_seq.fa", outpath))
    with open(script_path + "utility_files/similar_proteins/protein_info.pkl", "rb") as f:
        protein_info = pickle.load(f)

    results = "Query_ID\tUniProt_ID\tidentity\tProtein_Name\tGene_Name\tOrganism\tLength\n"
    with open(outpath + "similar_proteins_raw.tsv", "r") as f:
        lines = f.readlines()
    for line in lines:
        fields = line.strip().split()
        query_id = fields[0]
        subject_id = fields[1]
        identity = float(fields[2])
        if identity >= 50:
            Length, Entry_Name, Protein_Name, Gene_Name, Organism, Organism_ID = protein_info[subject_id]
            results += "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(query_id, subject_id, identity, Protein_Name, Gene_Name, Organism, Length)

    with open(outpath + "similar_proteins.tsv", "w") as f:
        f.write(results)
    os.system("rm " + outpath + "similar_proteins_raw.tsv")


def export_GPSite_predictions(predictions, id2seq, outpath):
    # original order: ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]
    thresholds = [0.25, 0.54, 0.42, 0.46, 0.69, 0.75, 0.63, 0.55, 0.36, 0.59] # select by maximizing MCC on the cross validation
    index = [2, 3, 1, 0, 8, 9, 4, 5, 6, 7] # switch order to ["DNA", "RNA", "PEP", "PRO", "ATP", "HEME", "ZN", "CA", "MG", "MN"]
    GPSite_binding_scores = {}

    for ID in predictions:
        seq = id2seq[ID][0:MAX_SEQ_LEN_ESMFold]
        preds = predictions[ID]
        norm_preds = []
        binding_scores = [] # protein-level binding scores

        for lig_idx, pred in enumerate(preds):
            threshold = thresholds[lig_idx]

            norm_pred = []
            for score in pred:
                if score > threshold:
                    norm_score = (score - threshold) / (1 - threshold) * 0.5 + 0.5
                else:
                    norm_score = (score / threshold) * 0.5
                norm_pred.append(norm_score)
            norm_preds.append(norm_pred)

            if lig_idx in [4, 5, 6, 7]: # metal ions
                k = 5
            else:
                k = 10
            k = min(k, len(seq))

            idx = np.argpartition(norm_pred, -k)[-k:]
            topk_norm_sores = np.array(norm_pred)[idx]
            binding_scores.append(topk_norm_sores.mean())

        GPSite_binding_scores[ID] = binding_scores


        pred_txt = "No.\tAA\tDNA_binding\tRNA_binding\tPeptide_binding\tProtein_binding\tATP_binding\tHEM_binding\tZN_binding\tCA_binding\tMG_binding\tMN_binding\n"
        for j in range(len(seq)):
            pred_txt += "{}\t{}".format(j+1, seq[j]) # 1-based

            for idx in index:
                norm_score = norm_preds[idx][j]
                pred_txt += "\t{:.3f}".format(norm_score)

            pred_txt += "\n"

        with open("{}/pred_GPSite/{}.txt".format(outpath, ID), "w") as f:
            f.write(pred_txt)


        # export the predictions to a pdb file (for the visualization in the server)
        score_lines = []
        for j in range(len(seq)):
            score_line = ""
            for idx in index:
                score = norm_preds[idx][j]
                score = "{:.2f}".format(score * 100)
                score = " " * (6 - len(score)) + score
                score_line += score
            score_lines.append(score_line)

        with open("{}/pdb/{}.pdb".format(outpath, ID), "r") as f:
            lines = f.readlines()

        current_pos = -1
        new_pdb = ""
        for line in lines:
            if line[0:4] != "ATOM":
                continue
            if int(line[22:26].strip()) != current_pos:
                current_pos = int(line[22:26].strip())
                score_line = score_lines.pop(0)
            new_line = line[0:60] + score_line + "           " + line.strip()[-1] + "  \n"
            new_pdb += new_line
        new_pdb += "TER\n"

        with open("{}/pred_GPSite/{}.pdb".format(outpath, ID), "w") as f:
            f.write(new_pdb)


    with open(outpath + "esmfold_pred.log", "r") as f:
        lines = f.readlines()

    info_dict = {}
    for line in lines:
        if "pLDDT" in line:
            ID_len, pLDDT, pTM = line.strip().split("|")[-1].strip().split(",")[0:3]
            ID = ID_len.strip().split()[3]
            length = ID_len.strip().split()[6]
            pLDDT = float(pLDDT.strip().split()[1])
            pTM = float(pTM.strip().split()[1])
            info_dict[ID] = [length, pLDDT, pTM]

    entry_info = "ID\tLength\tpLDDT\tpTM\tDNA_Binding\tRNA_Binding\tPeptide_Binding\tProtein_Binding\tATP_Binding\tHEM_Binding\tZN_Binding\tCA_Binding\tMG_Binding\tMN_Binding\n"
    for ID in predictions:
        Length, pLDDT, pTM = info_dict[ID]
        binding_scores = GPSite_binding_scores[ID]
        binding_scores = np.array(binding_scores)[index] # switch order to DNA, RNA, PEP ...
        entry = "{}\t{}\t{}\t{:.3f}".format(ID, Length, pLDDT, pTM)
        for score in binding_scores:
            entry += "\t{:.3f}".format(score)
        entry_info += (entry + "\n")

    with open("{}/pred_GPSite/overview.info".format(outpath), "w") as f:
        f.write(entry_info)

    os.system("rm {}/esmfold_pred.log".format(outpath))


def export_SPROF_GO_predictions(predictions, ID_list, success_ID, outpath, script_path):
    with open(script_path + "utility_files/SPROF-GO/idx_to_GO.pkl", "rb") as f:
        idx_to_GO = pickle.load(f)
    label_size = {"MF":790, "BP":4766, "CC":667}

    all_pred = "GO term id and name:\n"
    for task in ["MF", "BP", "CC"]:
        all_pred += (task + ":\n")
        all_pred += ("; ".join([term[0] for term in idx_to_GO[task]]) + "\n")
        all_pred += ("; ".join([term[1] for term in idx_to_GO[task]]) + "\n")
    all_pred += "\n"

    for i in range(len(ID_list)):
        if ID_list[i] not in success_ID:
            continue
        all_pred += (ID_list[i] + "\n")

        for j, task in enumerate(["MF", "BP", "CC"]):
            all_pred += (task + ":\n")
            all_pred += ("; ".join([str(score) for score in predictions[j][i]]) + "\n")
        all_pred += "\n"

    with open(outpath + "/pred_SPROF-GO/GO_preds.txt", "w") as f:
        f.write(all_pred)


def MakeFileForDownload(outpath):
    for dir_name in ["secondary_structure", "protein_binding", "GO_function", "subcellular_location", "solubility"]:
        os.makedirs(outpath + "/GPSFun_preds/" + dir_name, exist_ok = True)

    os.system("cp {}/DSSP/*.txt {}/GPSFun_preds/secondary_structure/".format(outpath, outpath))

    os.system("cp {}/pred_GPSite/* {}/GPSFun_preds/protein_binding/".format(outpath, outpath))
    for file in os.listdir("{}/GPSFun_preds/secondary_structure/".format(outpath)):
        ID = file[0:-7]
        with open("{}/GPSFun_preds/protein_binding/{}.pdb".format(outpath, ID), "r") as f:
            lines = f.readlines()
        lig = ["DNA", "RNA", "Peptide", "Protein", "ATP", "HEM", "ZN", "CA", "MG", "MN"]
        for k in range(10):
            pdb_single = ""
            for line in lines:
                if line[0:4] == "ATOM":
                    pdb_single += (line[0:60] + line[60+k*6:66+k*6] + line[120:])
            pdb_single += "TER\n"
            with open("{}/GPSFun_preds/protein_binding/{}_{}.pdb".format(outpath, ID, lig[k]), "w") as f:
                f.write(pdb_single)
        os.system("rm {}/GPSFun_preds/protein_binding/{}.pdb".format(outpath, ID))

    os.system("cp {}/pred_SPROF-GO/GO_preds.txt {}/GPSFun_preds/GO_function/".format(outpath, outpath))
    os.system("cp {}/pred_GPSLoc/SL_preds.csv {}/GPSFun_preds/subcellular_location/".format(outpath, outpath))
    os.system("cp {}/pred_GPSol/Sol_preds.csv {}/GPSFun_preds/solubility/".format(outpath, outpath))
    os.system("cp {}/similar_proteins.tsv {}/GPSFun_preds/".format(outpath, outpath))

    os.system("tar czvf {}/GPSFun_preds.tar.gz {}/GPSFun_preds".format(outpath, outpath))
