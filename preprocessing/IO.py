import pandas as pd

batch_size=4096
chunk_df=pd.read_csv('/mnt/isilon/wang_lab/shared/Belka/raw_data/train_split.csv', low_memory=False, chunksize = batch_size)

for i, batch_df in enumerate(chunk_df):
    #batch_df is the a pandas dataframe of size batch_size
    # do your processing or training here
    print(i, batch_df.shape)
