import re
import numpy as np
# import sys

def scores_generator(saving_path, args, data_pack, a = 0.5):
    if args.a_trade_off_flag:
        print('Start to compute scores with a trade-off, this may take some time')
        print('To cancel the a trade-off, you may set a_trade_flag to False')
    else:
        print('Start to compute scores')

    data_save_path = saving_path + '/eer_data_epoch' + str(args.epochs)
    probability_spk = np.load(data_save_path + 'spk_value.npy')
    probability_utt = np.load(data_save_path + 'utt_value.npy')
    target_te_0_numpy = np.load(data_save_path + 'spk_label.npy')
    target_te_1_numpy = np.load(data_save_path + 'utt_label.npy')
    # session_path = saving_path + '/test_'
    session = np.load(data_save_path + 'session.npy').astype(int)
    probability_spk *= a
    probability_utt *= (1 - a)
    num_labels_spk = probability_spk.shape[1]
    num_labels_utt = probability_utt.shape[1]

    probability_all = np.ones((num_labels_spk, num_labels_utt, 10, num_labels_spk, num_labels_utt),
                              dtype=np.float32)

    probability_spk = np.expand_dims(probability_spk, axis=2)
    probability_utt = np.expand_dims(probability_utt, axis=1)
    probability_spk = np.repeat(probability_spk, repeats=num_labels_utt, axis=2)
    probability_utt = np.repeat(probability_utt, repeats=num_labels_spk, axis=1)

    probability_utt_spk = probability_spk + probability_utt

    for i in range(probability_spk.shape[0]):
        probability_all[int(target_te_0_numpy[i]), int(target_te_1_numpy[i]), int(session[i]), :,
        :] = probability_utt_spk[i, :, :]

    trial_file_path = args.trails_path + 'part' + str(args.part) + r'/ndx/' + '3sess-pwd_' + str(args.dev_eval) + '_'
    if args.gender == 'male':
        trial_file_path += r'm.ndx'
    else:
        trial_file_path += r'f.ndx'
    trails_file = open(trial_file_path)
    line_count = 0

    # gen_c_scores = saving_path + '/epoch' + str(args.epochs) + 'gen_c_scores_trials'
    # gen_w_scores = saving_path + '/epoch' + str(args.epochs) + 'gen_w_scores_trials'
    # imp_c_scores = saving_path + '/epoch' + str(args.epochs) + 'imp_c_scores_trials'
    # imp_w_scores = saving_path + '/epoch' + str(args.epochs) + 'imp_w_scores_trials'
    gen_c_trials = saving_path + '/epoch' + str(args.epochs) + 'gen_c_perf_trials' + str(a)
    gen_w_trials = saving_path + '/epoch' + str(args.epochs) + 'gen_w_perf_trials' + str(a)
    imp_c_trials = saving_path + '/epoch' + str(args.epochs) + 'imp_c_perf_trials' + str(a)
    imp_w_trials = saving_path + '/epoch' + str(args.epochs) + 'imp_w_perf_trials' + str(a)

    if args.gender == 'male' and args.dev_eval == 'dev' and args.part == '1':
        total_trial_line = 7047582
    if args.gender == 'female' and args.dev_eval == 'dev' and args.part == '1':
        total_trial_line = 6251948
    if args.gender == 'male' and args.dev_eval == 'dev' and args.part == '2':
        total_trial_line = 7081815
    if args.gender == 'female' and args.dev_eval == 'dev' and args.part == '2':
        total_trial_line = 6281322
    if args.gender == 'male' and args.dev_eval == 'eval' and args.part == '1':
        total_trial_line = 9199116
    if args.gender == 'female' and args.dev_eval == 'eval' and args.part == '1':
        total_trial_line = 6818940
    if args.gender == 'male' and args.dev_eval == 'eval' and args.part == '2':
        total_trial_line = 9202704
    if args.gender == 'female' and args.dev_eval == 'eval' and args.part == '2':
        total_trial_line = 6822036

    while 1:
        line_count += 1
        if line_count % 500000 == 0:
            print('a = ' + str(a) + ': ' + str(line_count) + "/" + str(total_trial_line) + "(" + str(
                float(float(line_count) / float(total_trial_line)) * 100) + '%)')
        line = trails_file.readline()

        if not line:
            print('a = ' + str(a) + ': ' + str(total_trial_line) + "/" + str(total_trial_line) + "(" + str(
                float(float(total_trial_line) / float(total_trial_line)) * 100) + '%)')
            break
        else:
            if args.gender == 'male':  # male
                list_line = re.split('[m_/.,]', line)
            else:  # female
                list_line = re.split('[f_/.,]', line)

            model_spk = int(list_line[1]) - data_pack.STspkid
            model_utt = int(list_line[2]) - data_pack.STuttid
            test_spk = int(list_line[6]) - data_pack.STspkid
            test_utt = int(list_line[8]) - data_pack.STuttid
            test_ses = int(list_line[7])

            if model_spk == test_spk:
                if model_utt == test_utt:  # gen_c
                    # if probability_all[test_spk, test_utt, test_ses, model_spk, model_utt] < 0:
                    with open(gen_c_trials, 'a') as gen_c_t:
                        save_str = str(
                            probability_all[test_spk, test_utt, test_ses, model_spk, model_utt]) + ' target\n'
                        gen_c_t.write(save_str)
                        # with open(gen_c_scores, 'ab') as gen_c_s:
                        #     save_str = line + ' ' + str(
                        #         probability_all[test_spk, test_utt, test_ses, model_spk, model_utt]) + ' target\n'
                        #     gen_c_s.write(save_str)
                    # else:
                    #
                    #     print('=========================== ERROR!!!! =============================')
                    #     print(test_spk, test_utt, test_ses, model_spk, model_utt)
                    #     print('gen_c')
                else:  # gen_w
                    # if probability_all[test_spk, test_utt, test_ses, model_spk, model_utt] < 0:
                    with open(gen_w_trials, 'a') as gen_w_t:
                        save_str = str(probability_all[
                                           test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        gen_w_t.write(save_str)
                        # with open(gen_w_scores, 'ab') as gen_w_s:
                        #     save_str = line + ' ' + str(probability_all[
                        #                                     test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        #     gen_w_s.write(save_str)
                    # else:
                        # print('=========================== ERROR!!!! =============================')
                        # print(test_spk, test_utt, test_ses, model_spk, model_utt)
                        # print('gen_w')
            else:
                if model_utt == test_utt:  # imp_c
                    # if probability_all[test_spk, test_utt, test_ses, model_spk, model_utt] < 0:
                    with open(imp_c_trials, 'a') as imp_c_t:
                        save_str = str(probability_all[
                                           test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        imp_c_t.write(save_str)
                        # with open(imp_c_scores, 'ab') as imp_c_s:
                        #     save_str = line + ' ' + str(probability_all[
                        #                                     test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        #     imp_c_s.write(save_str)
                    # else:
                    #     print('=========================== ERROR!!!! =============================')
                    #     print(test_spk, test_utt, test_ses, model_spk, model_utt)
                    #     print('imp_c')
                else:  # imp_w
                    # if probability_all[test_spk, test_utt, test_ses, model_spk, model_utt] < 0:
                    with open(imp_w_trials, 'a') as imp_w_t:
                        save_str = str(probability_all[
                                           test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        imp_w_t.write(save_str)
                        # with open(imp_w_scores, 'ab') as imp_w_s:
                        #     save_str = line + ' ' + str(probability_all[
                        #                                     test_spk, test_utt, test_ses, model_spk, model_utt]) + ' nontarget\n'
                        #     imp_w_s.write(save_str)
                    # else:
                    #     print('=========================== ERROR!!!! =============================')
                    #     print(test_spk, test_utt, test_ses, model_spk, model_utt)
                    #     print('imp_w')

def scores(saving_path, args, data_pack):
    if args.a_trade_off_flag:
        for i in range(0, 21):
            a = float(i) / 20
            scores_generator(saving_path, args, data_pack, a=a)
    else:
        scores_generator(saving_path, args, data_pack, a=0.5)
    print('done')