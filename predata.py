import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}

def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="Convolution", drug_len=2048):

    #print("Parsing {0} , {1}, {2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))

    protein_col = "Protein_ID"
    drug_col = "Compound_ID"               #药物和蛋白的ID
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"                 
        col_names += [label_col]               #以上都是初始定义
    dti_df = dti_dir                    
    drug_df = pd.read_excel(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_excel(protein_dir, index_col="Protein_ID")
    

    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)
    drug_feature = np.stack(dti_df[drug_vec].map(lambda fp: fp.split(" ")))
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split(" ")))
    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        return {"protein_feature": protein_feature, "drug_feature": drug_feature, "label": label}
    else:
        return {"protein_feature": protein_feature, "drug_feature": drug_feature}
    
protein_data = "new_data/final_pro.xlsx"
drug_data = "new_data/final_drug.xlsx"
dti_train = pd.read_excel("new_data/final_dti.xlsx") 
data = parse_data(dti_train,drug_data,protein_data, with_label=True, prot_len=2500, prot_vec="Convolution",
                   drug_vec="morgan_fp_r2", drug_len=2048)

data_di =  { "predict" : data}
def shuju(**data):
    for dataset in data:
        protein = data[dataset]['protein_feature']
        drug = data[dataset]['drug_feature']
        drug = drug.astype(np.int16)
        label = data[dataset]['label']
        feature = np.hstack((protein,drug))
    return feature,label

feature,label = shuju(**data_di)
np.save("npy/feature.npy", feature.reshape(5019,4548))