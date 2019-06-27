import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class lstm(nn.Module):
    def __init__(self, data_pack, args):
        super(lstm, self).__init__()
        self.args = args
        self.input_dim_feat = data_pack.input_dim_feat
        self.num_labels_spk = data_pack.num_labels_spk
        self.num_labels_utt = data_pack.num_labels_utt
        self.max_fealen = data_pack.max_fealen
        # self.dropout_layer = nn.Dropout(p=0.2)
        self.batchnorm1d_spk = nn.BatchNorm1d(self.num_labels_spk)
        self.batchnorm1d_utt = nn.BatchNorm1d(self.num_labels_utt)

        self.lstm_all = nn.LSTM(input_size=self.input_dim_feat, hidden_size=self.args.dim_all, batch_first=True)
        self.lstm_utt = nn.LSTM(input_size=self.args.dim_all, hidden_size=self.args.dim_utt, batch_first=True)
        self.lstm_spk = nn.LSTM(input_size=self.args.dim_all, hidden_size=self.args.dim_spk, batch_first=True)

        self.fc_snf = nn.Linear(self.args.dim_spk, self.num_labels_spk)  # spk not full
        self.fc_unf = nn.Linear(self.args.dim_utt, self.num_labels_utt)  # utt not full

        # self.fc_both = nn.Linear(self.input_dim_feat * max_fealen, 1500)
        self.fc_spk = nn.Linear(self.args.dim_spk * self.max_fealen, self.num_labels_spk)
        self.fc_utt = nn.Linear(self.args.dim_utt * self.max_fealen, self.num_labels_utt)

    def forward_spk(self, batch, index, hidden_all, hidden_spk):  # speaker
        self.hidden_all = hidden_all
        self.hidden_spk = hidden_spk

        lstm_out_all, self.hidden_all = self.lstm_all(batch, self.hidden_all)
        lstm_out_spk, self.hidden_spk = self.lstm_spk(lstm_out_all, self.hidden_spk)

        dot_mul = torch.FloatTensor(batch.shape[0], self.max_fealen, self.args.dim_spk).fill_(0)
        for i in range(batch.shape[0]):
            dot_mul[i, 0:int(index[i]), :] = 1

        dot_mul = Variable(dot_mul).cuda()

        lstm_out_spk = lstm_out_spk * dot_mul

        output = self.fc_snf(lstm_out_spk)

        output_mean = torch.mean(output, dim=1, keepdim=True)

        output = torch.squeeze(output_mean, dim=1)

        output = self.batchnorm1d_spk(output)

        scores = F.log_softmax(output, dim=1)

        return scores

    def forward_utt(self, batch, index, hidden_all, hidden_utt):  # utterance
        self.hidden_all = hidden_all
        self.hidden_utt = hidden_utt

        lstm_out_all, self.hidden_all = self.lstm_all(batch, self.hidden_all)

        lstm_out_utt, self.hidden_utt = self.lstm_utt(lstm_out_all, self.hidden_utt)

        dot_mul = torch.FloatTensor(batch.shape[0], self.max_fealen, self.args.dim_utt).fill_(0)
        for i in range(batch.shape[0]):
            dot_mul[i, 0:int(index[i]), :] = 1

        dot_mul = Variable(dot_mul).cuda()

        lstm_out_utt = lstm_out_utt * dot_mul

        output = self.fc_unf(lstm_out_utt)

        output_mean = torch.mean(output, dim=1, keepdim=True)

        output = torch.squeeze(output_mean, dim=1)

        output = self.batchnorm1d_utt(output)

        scores = F.log_softmax(output, dim=1)

        return scores
