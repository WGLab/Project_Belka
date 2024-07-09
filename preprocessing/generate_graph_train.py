import numpy as np
import os, random, time
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
import multiprocessing as mp

# Atom Featurisation

def one_hot_encoding(x, permitted_list):
    """
    Maps input elements x which are not in the permitted list to the last element
    of the permitted list.
    """
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

#Main atom feat. func

def get_atom_features(atom, use_chirality=True):
    # Define a simplified list of atom types
    permitted_atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I','Dy', 'Unknown']
    atom_type = atom.GetSymbol() if atom.GetSymbol() in permitted_atom_types else 'Unknown'
    atom_type_enc = one_hot_encoding(atom_type, permitted_atom_types)
    
    # Consider only the most impactful features: atom degree and whether the atom is in a ring
    atom_degree = one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 'MoreThanFour'])
    is_in_ring = [int(atom.IsInRing())]
    
    # Optionally include chirality
    if use_chirality:
        chirality_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_features = atom_type_enc + atom_degree + is_in_ring + chirality_enc
    else:
        atom_features = atom_type_enc + atom_degree + is_in_ring
    
    return np.array(atom_features, dtype=np.float32)


# Bond featurization

def get_bond_features(bond):
    # Simplified list of bond types
    permitted_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC, 'Unknown']
    bond_type = bond.GetBondType() if bond.GetBondType() in permitted_bond_types else 'Unknown'
    
    # Features: Bond type, Is in a ring
    features = one_hot_encoding(bond_type, permitted_bond_types) \
               + [int(bond.IsInRing())]
    
    return np.array(features, dtype=np.float32)

# maxpool function for fast processing
def generate_graph_features_from_smiles(smiles, molecule_id, label):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = np.array(atom_features, dtype=np.float32)
    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # Undirected graph
        bond_feature = get_bond_features(bond)
        edge_features.append(bond_feature)
        edge_features.append(bond_feature)  # Same features in both directions
    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_attr = np.array(edge_features, dtype=np.float32)
    return x, edge_index, edge_attr, molecule_id, label

# Multiprocessing pool function

def process_chunk(df_chunk):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(generate_graph_features_from_smiles, zip(df_chunk['molecule_smiles'], df_chunk['id'], df_chunk['binds']))
    return results

# Convert results to Data objects
def convert_to_data_objects(results):
    data_list = []
    for result in results:
        if result is not None:
            x, edge_index, edge_attr, molecule_id, label = result
            data = Data(x=torch.tensor(x, dtype=torch.float),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                        y=torch.tensor([label], dtype=torch.float),
                        molecule_id=torch.tensor([molecule_id], dtype=torch.long))
            data_list.append(data)
    return data_list


def downsample_data(df, positive_label=1, negative_label=0, ratio=5):
    positive_samples = df[df['binds'] == positive_label]
    negative_samples = df[df['binds'] == negative_label]
    downsampled_negative_samples = negative_samples.sample(n=len(positive_samples) * ratio, random_state=42)
    balanced_df = pd.concat([positive_samples, downsampled_negative_samples])
    return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

#########################################################################################
########## MODIFY HERE: designate data type, batch size, and the file location###########
num_of_cpu = 48
type_data = 'train_downsample' # train / test / validation
chunk_size = 1000000  #load 1M rows for each batch (assume ~100k rows for each protein)
chunk_df = pd.read_csv(f'/mnt/isilon/wang_lab/shared/Belka/raw_data/shuffled_raw_data/train_split.csv', low_memory=False, chunksize=chunk_size)
save_parent_path = '/mnt/isilon/wang_lab/shared/Belka/analysis/feature_generate/graph_features/'
#########################################################################################

##### below are all automation, no need to change #####
# define map for protein automation
p_list = ['seh','hsa','brd4']
p_map = {'seh':'sEH','hsa':'HSA','brd4':'BRD4'}
for p in p_list:
    save_dir=save_parent_path+f'{type_data}/{p}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
brd4_save_dir = save_parent_path+f'{type_data}/brd4/'
hsa_save_dir = save_parent_path+f'{type_data}/hsa/'
seh_save_dir = save_parent_path+f'{type_data}/seh/'
p_save_map = {'seh':seh_save_dir,'hsa':hsa_save_dir,'brd4':brd4_save_dir}


for i, batch_df in enumerate(chunk_df):
    batch_df = downsample_data(batch_df, ratio=5)
    
    with mp.Pool(num_of_cpu) as pool:  
        for protein in p_list:
            df = batch_df[batch_df['protein_name']==p_map.get(protein)]

            results = process_chunk(df)
            data_list = convert_to_data_objects(results)
            torch.save(data_list, f'{p_save_map.get(protein)}data{i}.pt')
            print(f'{protein}-{i}')

print(f'{type_data} graph feature generation finished.')