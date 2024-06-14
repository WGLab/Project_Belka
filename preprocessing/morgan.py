import pandas as pd
import numpy as np
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem

# Generate ECFPs
def generate_ecfp(molecule):
    radius=2
    bits=1024 #!!we could change radius and bits
    if molecule is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(molecule), radius, nBits=bits))

p_map={'BRD4':0, 'HSA':1, 'sEH':2}
batch_size=500000
#change input dir to validation_split.csv for validation dataset
chunk_df=pd.read_csv('/mnt/isilon/wang_lab/shared/Belka/raw_data/train_split.csv', low_memory=False, chunksize = batch_size)

for i, batch_df in enumerate(chunk_df):
    with mp.Pool(24) as pool:    
        results=pool.map(generate_ecfp, batch_df['molecule_smiles'])
        morgan=np.vstack(results).astype(np.int8)
        protein=np.eye(3)[[p_map[x] for x in batch_df['protein_name']]].astype(np.int8)
        labels=np.array(batch_df.binds).astype(np.int8)
        #change save dir to morgan_validation for validation_split.csv
        np.savez('morgan/data{}.npz'.format(i), morgan=morgan, protein=protein, labels=labels)
        print(i)
