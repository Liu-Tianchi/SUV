import sys
import os
import torch

def pre_check(args):
    # check GPU
    print(' ----------------------------------------------------------- ')
    print(' ------------ Start Checking and Create Folder ------------- ')

    if not torch.cuda.is_available():
        print(' ----------------------------------------------------------- ')
        print(' --------------------- GPU is not available ---------------- ')
        print(' ----------------------------------------------------------- ')
        sys.exit(0)
    # check folder to save
    count = 1
    while True:
        path = os.getcwd() + '/' + args.dev_eval + '_p' + args.part + '_' + args.gender + '_' + str(count)
        folder_exists = os.path.exists(path)
        if not folder_exists:
            os.makedirs(path)
            break
        else:
            count += 1

    print(' ------------------------- Done! --------------------------- ')
    print(' ----------------------------------------------------------- ')

    return path