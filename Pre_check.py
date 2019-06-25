import sys
import os
import torch

def pre_check(args):
    # check some input information
    print(' ----------------------------------------------------------- ')
    print(' -------------- Start Checking before running -------------- ')
    # # gender
    # if args.gender != 'male' and args.gender != 'female':
    #     print(' ----------------------------------------------------------- ')
    #     print(' ---------------- WRONG GENDER INFORMATION ----------------- ')
    #     print(' ------------- Input Should be male or female -------------- ')
    #     print(' ----------------------------------------------------------- ')
    #     sys.exit(0)
    #
    # # check dataset
    # if args.dev_eval != 'dev' and args.dev_eval != 'eval':
    #     print(' ----------------------------------------------------------- ')
    #     print(' ---------------- WRONG DEV_EVAL INFORMATION --------------- ')
    #     print(' ---------------- Input Should be dev or eval -------------- ')
    #     print(' ----------------------------------------------------------- ')
    #     sys.exit(0)
    #
    # # check part number
    # if args.part != '1' and args.part != '2':
    #     print(' ----------------------------------------------------------- ')
    #     print(' ------------------- WRONG PART INFORMATION ---------------- ')
    #     print(' ------------------- Input Should be 1 or 2 ---------------- ')
    #     print(' ----------------------------------------------------------- ')
    #     sys.exit(0)

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