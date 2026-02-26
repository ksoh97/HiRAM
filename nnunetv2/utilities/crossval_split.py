import os

def generate_crossval_split(txt_path):
    with open(os.path.join(txt_path, "trn_list.txt"), "r") as f:
        trn_list = [line.strip() for line in f.readlines()]
    with open(os.path.join(txt_path, "tst_list.txt"), "r") as f:
        val_list = [line.strip() for line in f.readlines()]
    trn_list = [os.path.basename(x).replace('.nii.gz', '') for x in trn_list]
    val_list = [os.path.basename(x).replace('.nii.gz', '') for x in val_list]

    splits = {}
    splits['train'] = [name.split('/')[-1] for name in trn_list]
    splits['val'] = [name.split('/')[-1] for name in val_list]

    return splits
