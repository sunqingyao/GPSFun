# Introduction
GPSFun is a Geometry-aware Protein Sequence Function annotation toolkit, which is fast, accurate and versatile. Simply requiring a sequence as input, GPSFun can provide comprehensive predictions including: (1) protein secondary structure and 3D conformation; (2) protein binding residues with DNA, RNA, peptide, protein, ATP, HEM, and metal ions; (3) protein function defined by Gene Ontology (GO); (4) protein subcellular location; and (5) protein solubility. By leveraging large language models, GPSFun is liberated from the reliance on MSA or experimental protein structures, thus enabling fast and accurate predictions from sequences. Experiments demonstrate that GPSFun substantially surpasses state-of-the-art sequence-based and structure-based approaches on various tasks. If your input is small, you are recommended to use our [GPSFun webserver](https://bio-web1.nscc-gz.cn/app/GPSFun). 

# System requirement
GPSFun is mainly based on the following packages:  
- python  3.8.16  
- numpy  1.24.3  
- pytorch  1.13.1  
- pytorch-scatter  2.1.1  
- pytorch-cluster  1.6.1  
- pyg  2.3.0  
- biopython  1.81  
- fair-esm  2.0.0  
- dllogger  1.0.0  
- openfold  1.0.1  
- sentencepiece  0.1.99  
- transformers  4.30.1

While we have not tested other versions, any reasonably recent versions of these requirements should work.

# Install and set up GPSFun
**1.** Download the code in ZIP archive.  
**2.** Install the packages required by GPSFun. To install [ESMFold](https://github.com/facebookresearch/esm) and [ProtTrans](https://github.com/agemagician/ProtTrans), one can follow their official tutorials. However, at the time of writing, we found some commands in the installation tutorial of ESMFold didn't work. To avoid unnecessary troubles, you can install GPSFun according to the following instructions that we cleaned up:  
**a.** To use the ESMFold model, make sure you start from an environment with python <= 3.9 and pytorch installed. Here we use python 3.8.16 and pytorch 1.13.1 with cuda version of 11.6. Then run:
```
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install modelcif==0.7
```
Finally, download the [openfold](https://github.com/aqlaboratory/openfold) ZIP archive, unzip it, and run `python3 setup.py install`  
**b.** To use the ProtTrans model, run:
```
pip install SentencePiece transformers
```
**c.** Finally, install the remaining packages required by GPSFun:
```
conda install pyg -c pyg
conda install pytorch-scatter -c pyg
conda install pytorch-cluster -c pyg
```
**3.** Download the pre-trained ProtT5-XL-UniRef50 model in [here](https://zenodo.org/record/4644188) (~ 5.3 GB). The ESM2 and ESMFold models will be automatically downloaded at the first time you run GPSFun, which will take some time (~ 7.9 GB).  
**4.** Set the path variable `ProtTrans_path` in `./script/predict.py`  
**5.** Add permission to execute for DSSP by `chmod +x ./script/feature_extraction/mkdssp`  

# Run GPSFun for prediction
Simply run the following command to predict the functions of the sequences in `demo.fa` on GPU (id = 0):
```
python ./script/predict.py -f ./example/demo.fa -o ./example/ --gpu 0
```
Omitting the `--gpu` parameter will make GPSFun run on CPU, which will take more time especially for the structure prediction pipeline. The prediction results will be saved under `./example/`. Here we provide the corresponding canonical input and prediction results under `./example/` for your reference.

# Datasets
We provide the datasets for training and evaluating GPSFun under `./datasets/` .  For the protein binding site datasets, each protein is associated with 3 lines of protein name, sequence and residue labels (0 for non-binding and 1 for binding). For the GO datasets, we provide the GO term and DAG definitions in an obo file.

# Contact
Qianmu Yuan (yuanqm3@mail2.sysu.edu.cn)  
Yuedong Yang (yangyd25@mail.sysu.edu.cn)
