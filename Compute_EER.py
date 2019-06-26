import os

def compute_EER(saving_path, args):
    # use the compute-eer from kaldi
    if args.a_trade_off_flag:
        for i in range(0, 21):
            a = float(i) / 20
            cmd = r'./compute_eer_SUV.sh ' + saving_path + ' ' + str(args.epochs) + ' ' + str(a)
            os.system(cmd)
    else:
        cmd = r'./compute_eer_SUV.sh ' + saving_path + ' ' + str(args.epochs) + ' 0.5'
        os.system(cmd)