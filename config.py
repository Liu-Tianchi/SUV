import argparse

def parse_args():
     parser = argparse.ArgumentParser(
         description='Speaker-Utterance Verification Framework')
     # data set arguments
     parser.add_argument('--gender', default='male', type=str,
                         choices=['male', 'female'],
                         help='gender of data set')
     parser.add_argument('--part', default='1', type=str,
                         choices=['1', '2'],
                         help='part of data set')
     parser.add_argument('--dev_eval', default='dev', type=str,
                         choices=['dev', 'eval'],
                         help='development or evaluation part of data set')
     parser.add_argument('--feature_path', default=r'/home/tianchi/Desktop/kaldi/egs/rsr_system_transfree/',
                         help='the path to the feature and data folder of RSR2015 database')

     parser.add_argument('--trails_path', default=r'/home/tianchi/database/RSR2015/key/',
                         help='the path to the trails folder of RSR2015 database')
     parser.add_argument('--kaldi_io_path', default=r'/home/tianchi/Software/kaldi-io/')

     parser.add_argument('--demo', default=False,
                         help='whether to run the demo set by author')

     # training arguments
     parser.add_argument('--epochs', default=1500, type=int,
                         help='total epochs to run')
     parser.add_argument('--batch_size', default=128, type=int,
                         help='batch_size for both training and testing')
     # parser.add_argument('--GPU_avaiable', default=True,
     #                     help='whether use GPU')

     # model arguments
     parser.add_argument('--dim_all', default=256, type=int,
                         help='hidden dimension of first layer LSTM')
     parser.add_argument('--dim_spk', default=256, type=int,
                         help='hidden dimension of spk LSTM')
     parser.add_argument('--dim_utt', default=256, type=int,
                         help='hidden dimension of utt LSTM')
     parser.add_argument('--load_model', default=False,
                         help='whether to load model')
     parser.add_argument('--model_path',
                         help='the path of the model that you want to load')

     # scores arguments
     parser.add_argument('--a_trade_off_flag', default=False,
                         help='whether to activate a_trade_off, if activate, it may cost more time before compute eer')

     args = parser.parse_args()

     return args
