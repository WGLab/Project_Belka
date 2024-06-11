with open('/mnt/isilon/wang_lab/shared/Belka/train.csv','r') as infile:
    with open('/mnt/isilon/wang_lab/shared/Belka/raw_data/train_split.csv','w') as trainfile, open('/mnt/isilon/wang_lab/shared/Belka/raw_data/validation_split.csv','w') as valfile:
        header=infile.readline()
        trainfile.write(header)
        valfile.write(header)
        for line in infile:
            if random.randint(1,10)<=8:
                trainfile.write(line)
            else:
                valfile.write(line)
