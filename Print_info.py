from datetime import datetime


def print_info(saving_path, args, data_pack):

    # datetime object containing current date and time
    now = datetime.now()
     
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    print()
    print(' ----------------------------------------------------------- ')
    print(' -------------------- Running Information ------------------ ')
    print('         Training Start from: ' + dt_string)	
    print('              Folder to save: ' + saving_path)
    if args.load_model:
        print('             Model load from: ' + args.model_path)
    print('             Running dataset: RSR2015 ' + args.dev_eval + '_p' + args.part + '_' + args.gender)
    print('                  Batch size: ' + str(args.batch_size))
    print('         Total epochs to run: ' + str(args.epochs))
    print('        Total dim of feature: ' + str(data_pack.input_dim_feat))
    print('           Number of Speaker: ' + str(data_pack.num_labels_spk))
    print('         Number of Utterance: ' + str(data_pack.num_labels_utt))
    print('        Start num of Speaker: ' + str(data_pack.STspkid))
    print('      Start num of Utterance: ' + str(data_pack.STuttid))
    print(' Hid dim of shared, spk, utt: ' + str(args.dim_all) + ', ' + str(args.dim_spk) + ', ' + str(args.dim_utt))
    print(' ----------------------------------------------------------- ')
    
    with open(saving_path + "/Running_info.txt", "w") as text_file:
      print(' ----------------------------------------------------------- ', file=text_file)
      print(' -------------------- Running Information ------------------ ', file=text_file)
      print('         Training Start from: ' + dt_string, file=text_file)
      print('              Folder to save: ' + saving_path, file=text_file)
      if args.load_model:
          print('             Model load from: ' + args.model_path, file=text_file)
      print('             Running dataset: RSR2015 ' + args.dev_eval + '_p' + args.part + '_' + args.gender, file=text_file)
      print('                  Batch size: ' + str(args.batch_size), file=text_file)
      print('         Total epochs to run: ' + str(args.epochs), file=text_file)
      print('        Total dim of feature: ' + str(data_pack.input_dim_feat), file=text_file)
      print('           Number of Speaker: ' + str(data_pack.num_labels_spk), file=text_file)
      print('         Number of Utterance: ' + str(data_pack.num_labels_utt), file=text_file)
      print('        Start num of Speaker: ' + str(data_pack.STspkid), file=text_file)
      print('      Start num of Utterance: ' + str(data_pack.STuttid), file=text_file)
      print(' Hid dim of shared, spk, utt: ' + str(args.dim_all) + ', ' + str(args.dim_spk) + ', ' + str(args.dim_utt), file=text_file)
      print(' ----------------------------------------------------------- ', file=text_file)
