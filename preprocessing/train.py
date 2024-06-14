import numpy as np
import os, random
from pathlib import Path

def get_train_files(input_list):
    out_list=[]
    if type(input_list)==type(None):
        return out_list
    for item in input_list:
        
        if os.path.isdir(item):
            out_list.extend(list(Path(item).rglob("*.npz")))

        elif item[-4:]=='.npz':
            out_list.append(item)        
    
    random.seed(0)
    random.shuffle(out_list)
    return out_list

def read_from_file(file_name):
    data=np.load(file_name)
    return data['morgan'], data['protein'], data['labels']

def generate_batches(files, batch_size=1024):
    counter = 0
    
    print_freq=max(1, len(files)//10)
    
    while counter<len(files):
        file_name = files[counter]

        counter +=1
        
        data=read_from_file(file_name)

        morgan, protein, labels=data
        batch_size=max(batch_size,1)
        for local_index in range(0, labels.shape[0], batch_size):
            batch_morgan=morgan[local_index:(local_index + batch_size)]
            batch_protein=protein[local_index:(local_index + batch_size)]
            batch_labels=labels[local_index:(local_index + batch_size)]          

            yield batch_morgan, batch_protein, batch_labels
        
        if counter%print_freq==0:
            print('.', end='',flush=True)
            

            
#load the files here
train_files=get_files(['morgan/'])
test_files=get_files(['morgan_validation/'])

# Training loop
for i in range(epoch_num):
    #train the model
    net.train() #initiate the model, change to your model
    train_loss = 0
    train_generator = generate_batches(train_files, batch_size=1024)
    for batch in train_generator:
        batch_morgan, batch_protein, batch_labels = batch
        batch_morgan = np.concatenate((batch_morgan, batch_protein), 1) #incorperate protein name: 1024+3=1027 features
        
        #train your model here (for NN example, check the 'train_by_trunk_NN' notebook.)
        ####train(xxx)

    with torch.no_grad():
        #test the model
        net.eval() #change this to your own model
        test_loss = 0
        test_generator = generate_batches(test_files, batch_size=1024)
        for batch in test_generator:
            batch_morgan, batch_protein, batch_labels = batch
            batch_morgan = np.concatenate((batch_morgan, batch_protein), 1)
            
            #test your model from here
            ####eval(xxx)

            
        # Calculate metrics here
        ap = average_precision_score(labels, preds)
        

        print(f'\nEpoch={i+1} Train_Loss={train_loss/len(labels):.6f} Test_Loss={test_loss/len(labels):.6f} '
              f'Test AP={ap:.6f}')
