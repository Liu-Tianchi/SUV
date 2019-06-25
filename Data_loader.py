

import numpy as np
import os
# import sys
import pickle



def STuttid_STspkid_numspk_numutt(args):
    if args.gender == 'male' and args.dev_eval == 'dev' and args.part == '1':  # dev_p1_male
        STspkid = 51
        STuttid = 1
        num_labels_spk = 50
    if args.gender == 'male' and args.dev_eval == 'dev' and args.part == '2':  # dev_p2_male
        STspkid = 51
        STuttid = 31
        num_labels_spk = 50
    if args.gender == 'male' and args.dev_eval == 'eval' and args.part == '1':  # eval_p1_male
        STspkid = 101
        STuttid = 1
        num_labels_spk = 57
    if args.gender == 'male' and args.dev_eval == 'eval' and args.part == '2':  # eval_p2_male
        STspkid = 101
        STuttid = 31
        num_labels_spk = 57
    if args.gender == 'female' and args.dev_eval == 'dev' and args.part == '1':  # dev_p1_female
        STspkid = 48
        STuttid = 1
        num_labels_spk = 47
    if args.gender == 'female' and args.dev_eval == 'dev' and args.part == '2':  # dev_p2_female
        STspkid = 48
        STuttid = 31
        num_labels_spk = 47
    if args.gender == 'female' and args.dev_eval == 'eval' and args.part == '1':  # eval_p1_female
        STspkid = 95
        STuttid = 1
        num_labels_spk = 49
    if args.gender == 'female' and args.dev_eval == 'eval' and args.part == '2':  # eval_p2_female
        STspkid = 95
        STuttid = 31
        num_labels_spk = 49
    return STspkid, STuttid, num_labels_spk

def data_loader(args):

    cmd = r'export PATH=$PATH:' + args.kaldi_io_path
    os.system(cmd)

    print()
    print(' ----------------------------------------------------------- ')
    STspkid, STuttid, num_labels_spk = STuttid_STspkid_numspk_numutt(args)
    if args.demo == True:

        print(' -------------- Load data via demo folder ------------------ ')
        with open('./demo_data/fea_tr_dev_p1_m.pickle', 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            fea_tr = u.load()
        with open('./demo_data/fea_te_dev_p1_m.pickle', 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            fea_te = u.load()
    else:
        import kaldi_io
        print(' ----------------- Load data via kaldi-io ------------------ ')

        path_data_folder = args.feature_path + args.gender + '/data/' + args.dev_eval + '_p' + args.part
        read_kaldi_command_tr = r'ark:copy-feats scp:' + path_data_folder + r'_tr/feats.scp ark:- | apply-cmvn  --norm-vars=true --utt2spk=ark:' + path_data_folder + r'_tr/utt2spk scp:' + path_data_folder + r'_tr/cmvn.scp ark:- ark:- | add-deltas ark:- ark:- |'
        read_kaldi_command_te = r'ark:copy-feats scp:' + path_data_folder + r'_te/feats.scp ark:- | apply-cmvn  --norm-vars=true --utt2spk=ark:' + path_data_folder + r'_te/utt2spk scp:' + path_data_folder + r'_te/cmvn.scp ark:- ark:- | add-deltas ark:- ark:- |'

        fea_tr = {k: m for k, m in kaldi_io.read_mat_ark(read_kaldi_command_tr)}
        fea_te = {k: m for k, m in kaldi_io.read_mat_ark(read_kaldi_command_te)}
    print(' ------------------------ Done! ---------------------------- ')
    print(' ----------------------------------------------------------- ')
    
    max_fealen = 0  # max length of feature
    # min_fealen = 10000
    fea_tr_key_list = list(fea_tr.keys())
    fea_te_key_list = list(fea_te.keys())
    num_tr_samples = len(fea_tr_key_list)
    num_te_samples = len(fea_te_key_list)
    input_dim_feat = fea_tr[fea_tr_key_list[0]].shape[1]  # dim of feature

    index_tr = np.zeros(num_tr_samples, dtype=np.int32)
    index_te = np.zeros(num_te_samples, dtype=np.int32)

    count = 0
    for k in range(num_tr_samples):
        max_fealen = max(max_fealen, fea_tr[fea_tr_key_list[k]].shape[0])
        # min_fealen = min(min_fealen, fea_tr[fea_tr_key_list[k]].shape[0])
        index_tr[count] = fea_tr[fea_tr_key_list[k]].shape[0]
        count += 1

    count = 0
    for k in range(num_te_samples):
        max_fealen = max(max_fealen, fea_te[fea_te_key_list[k]].shape[0])
        # min_fealen = min(min_fealen, fea_te[fea_te_key_list[k]].shape[0])
        index_te[count] = fea_te[fea_te_key_list[k]].shape[0]
        count += 1

    xtr = np.zeros((num_tr_samples, max_fealen, input_dim_feat), dtype=np.float32)
    ytr_spk = np.zeros(num_tr_samples, dtype=np.int32)
    ytr_utt = np.zeros(num_tr_samples, dtype=np.int32)

    xte = np.zeros((num_te_samples, max_fealen, input_dim_feat), dtype=np.float32)
    yte_spk = np.zeros(num_te_samples, dtype=np.int32)
    yte_utt = np.zeros(num_te_samples, dtype=np.int32)
    yte_ses = np.zeros(num_te_samples, dtype=np.int32)  # session

    for k in range(num_tr_samples):
        # xtr.append(fea_tr[fea_tr_key_list[k]])  # list
        nfrm = fea_tr[fea_tr_key_list[k]].shape[0]
        xtr[k, 0:nfrm, :] = fea_tr[fea_tr_key_list[k]]
        ytr_spk[k] = int(fea_tr_key_list[k].split("_")[0][1:]) - STspkid  
        ytr_utt[k] = int(fea_tr_key_list[k].split("_")[-1]) - STuttid 

    for k in range(num_te_samples):
        nfrm = fea_te[fea_te_key_list[k]].shape[0]
        xte[k, 0:nfrm, :] = fea_te[fea_te_key_list[k]]
        # xte.append(fea_te[fea_te_key_list[k]])  # list
        yte_spk[k] = int(fea_te_key_list[k].split("_")[0][1:]) - STspkid 
        yte_utt[k] = int(fea_te_key_list[k].split("_")[-1]) - STuttid  
        yte_ses[k] = int(fea_te_key_list[k].split("_")[1])
    
    return STspkid, STuttid, num_labels_spk, xtr, ytr_spk, ytr_utt, xte, yte_spk, yte_utt, yte_ses, max_fealen, \
           fea_tr_key_list, fea_te_key_list, index_tr, index_te, input_dim_feat

class data_packed():

    def __init__(self):
        pass

    def data_process(self, args):

        self.STspkid, self.STuttid, self.num_labels_spk, self.xtr, self.ytr_spk, self.ytr_utt, self.xte, self.yte_spk, \
        self.yte_utt, self.yte_ses, self.max_fealen, self.fea_tr_key_list, self.fea_te_key_list, self.index_tr, \
        self.index_te, self.input_dim_feat = data_loader(args)

        self.num_labels_utt = 30
        # self.STspkid = STspkid
        # self.STuttid = STuttid
        # self.num_labels_spk = num_labels_spk
        # self.xtr = xtr


