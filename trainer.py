import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Trainer():
    def __init__(self, data_pack, saving_path, args, model, optimizer):
        super(Trainer, self).__init__()
        self.args = args
        self.model =model
        self.data_pack = data_pack
        self.saving_path = saving_path
        self.optimizer = optimizer

    def init_hidden(self, batch_size, hidden_d):
        # if self.args.GPU_avaiable:
        #     return (Variable(torch.zeros(1, batch_size, hidden_d)).cuda(),
        #             Variable(torch.zeros(1, batch_size, hidden_d)).cuda())
        # else:
        #     return (Variable(torch.zeros(1, batch_size, hidden_d)),
        #             Variable(torch.zeros(1, batch_size, hidden_d)))
        return (Variable(torch.zeros(1, batch_size, hidden_d)).cuda(),
                Variable(torch.zeros(1, batch_size, hidden_d)).cuda())

    def train(self):
        print()
        print(' ----------------------------------------------------------- ')
        print(' --------------------- Start Training ---------------------- ')
        print()
        for epoch in range(1, self.args.epochs+1):
            start_time = time.time()
    
            print("Training for Epoch : " + str(epoch))
    
            self.model.train()

            randseq = np.random.permutation(len(self.data_pack.fea_tr_key_list))

            cnt = 0
            total_loss_spk = 0.0
            total_loss_utt = 0.0
    
            for batch_idx in range(int(len(self.data_pack.fea_tr_key_list) // self.args.batch_size)):  # do not train the last batch

                data = torch.from_numpy(self.data_pack.xtr[randseq[cnt: cnt + self.args.batch_size]])
                # if self.args.GPU_avaiable:
                #     data = Variable(data).type(torch.FloatTensor).cuda()
                # else:
                #     data = Variable(data).type(torch.FloatTensor)
                data = Variable(data).type(torch.FloatTensor).cuda()
                # utt_task:
                self.hidden_all = self.init_hidden(data.shape[0], self.args.dim_all)
                self.hidden_utt = self.init_hidden(data.shape[0], self.args.dim_utt)

                index_tr_sub = self.data_pack.index_tr[randseq[cnt: cnt + data.shape[0]]]
                
                target_utt = torch.from_numpy(self.data_pack.ytr_utt[randseq[cnt: cnt + data.shape[0]]])
                # if self.args.GPU_avaiable:
                #     target_utt = Variable(target_utt).type(torch.LongTensor).cuda()
                # else:
                #     target_utt = Variable(target_utt).type(torch.LongTensor)
                target_utt = Variable(target_utt).type(torch.LongTensor).cuda()
                self.optimizer.zero_grad()
    
                output_utt = self.model.forward_utt(data, index_tr_sub, self.hidden_all, self.hidden_utt)
    
                loss_utt = F.nll_loss(output_utt, target_utt)
                total_loss_utt += float(loss_utt)
                loss_utt.backward()
                self.optimizer.step()
    
                # spk_task:
                self.hidden_all = self.init_hidden(data.shape[0], self.args.dim_all)
                self.hidden_spk = self.init_hidden(data.shape[0], self.args.dim_spk)

                index_tr_sub = self.data_pack.index_tr[randseq[cnt: cnt + data.shape[0]]]
                
                target_spk = torch.from_numpy(self.data_pack.ytr_spk[randseq[cnt: cnt + data.shape[0]]])
                # if self.args.GPU_avaiable:
                #     target_spk = Variable(target_spk).type(torch.LongTensor).cuda()
                # else:
                #     target_spk = Variable(target_spk).type(torch.LongTensor)
                target_spk = Variable(target_spk).type(torch.LongTensor).cuda()
                self.optimizer.zero_grad()

                output_spk = self.model.forward_spk(data, index_tr_sub, self.hidden_all, self.hidden_spk)
                loss_spk = F.nll_loss(output_spk, target_spk)
                total_loss_spk += float(loss_spk)
                loss_spk.backward()
                self.optimizer.step()

                # print
                cnt += self.args.batch_size
    
                if batch_idx % 5 == 0:

                    already_train_num = batch_idx * len(data)

                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss_spk: {:.6f} \tloss_utt: {:.6f}'.format(epoch,
                                                                                                          already_train_num,
                                                                                                          len(
                                                                                                              self.data_pack.fea_tr_key_list),
                                                                                                          100. * batch_idx * len(
                                                                                                              data) / len(
                                                                                                              self.data_pack.fea_tr_key_list),
                                                                                                          float(
                                                                                                              loss_spk.data),
                                                                                                          float(
                                                                                                              loss_utt.data)))

            average_loss_spk = float(total_loss_spk) / float(batch_idx + 1.0)
            average_loss_utt = float(total_loss_utt) / float(batch_idx + 1.0)
            print('Average loss: spk: ' + str(average_loss_spk) + ', utt: ' + str(average_loss_utt))
    
            train_time = time.time()
            print("Training time : " + str(train_time - start_time) + "second")
            print('########################################################################################\n')
        # print('')
    
    def test(self):

        print('Start testing for epoch %d' % self.args.epochs)
        self.model.eval()
        correct_spk = 0
        correct_utt = 0
        cnt = 0

        for batch_idx in range(int(len(self.data_pack.fea_te_key_list) // self.args.batch_size) + 1):
            # ses = yte_ses[cnt:min(len(fea_te_key_list), cnt + batch_size)]

            data = torch.from_numpy(self.data_pack.xte[cnt: min(len(self.data_pack.fea_te_key_list), cnt + self.args.batch_size)])
            # if self.args.GPU_avaiable:
            #     data = Variable(data).type(torch.FloatTensor).cuda()
            # else:
            #     data = Variable(data).type(torch.FloatTensor)
            data = Variable(data).type(torch.FloatTensor).cuda()

            # spk_task:
            self.hidden_all = self.init_hidden(data.shape[0], self.args.dim_all)
            self.hidden_spk = self.init_hidden(data.shape[0], self.args.dim_utt)
            index_te_sub = self.data_pack.index_te[cnt:min(len(self.data_pack.fea_te_key_list), cnt + self.args.batch_size)]

            target_te_spk_numpy = self.data_pack.yte_spk[cnt:min(len(self.data_pack.fea_te_key_list), cnt + self.args.batch_size)]

            target_te_spk = torch.from_numpy(target_te_spk_numpy)
            # if self.args.GPU_avaiable:
            #     target_te_spk = Variable(target_te_spk).type(torch.LongTensor).cuda()
            # else:
            #     target_te_spk = Variable(target_te_spk).type(torch.LongTensor)
            target_te_spk = Variable(target_te_spk).type(torch.LongTensor).cuda()
            output_spk = self.model.forward_spk(data, index_te_sub, self.hidden_all, self.hidden_spk)
            pred = output_spk.data.max(1, keepdim=True)[1]
            # if self.args.GPU_avaiable:
            #     correct_spk += pred.eq(target_te_spk.data.view_as(pred)).cuda().sum()
            # else:
            #     correct_spk += pred.eq(target_te_spk.data.view_as(pred)).sum()
            correct_spk += pred.eq(target_te_spk.data.view_as(pred)).cuda().sum()

            # utt_task:
            self.hidden_all = self.init_hidden(data.shape[0], self.args.dim_all)
            self.hidden_utt = self.init_hidden(data.shape[0], self.args.dim_utt)
            index_te_sub = self.data_pack.index_te[cnt:min(len(self.data_pack.fea_te_key_list), cnt + self.args.batch_size)]
            target_te_utt_numpy = self.data_pack.yte_utt[cnt:min(len(self.data_pack.fea_te_key_list), cnt + self.args.batch_size)]
            target_te_utt = torch.from_numpy(target_te_utt_numpy)

            # if self.args.GPU_avaiable:
            #     target_te_utt = Variable(target_te_utt).type(torch.LongTensor).cuda()
            # else:
            #     target_te_utt = Variable(target_te_utt).type(torch.LongTensor)
            target_te_utt = Variable(target_te_utt).type(torch.LongTensor).cuda()
            output_utt = self.model.forward_utt(data, index_te_sub, self.hidden_all, self.hidden_utt)
            pred = output_utt.data.max(1, keepdim=True)[1]

            # if self.args.GPU_avaiable:
            #     correct_utt += pred.eq(target_te_utt.data.view_as(pred)).cuda().sum()
            # else:
            #     correct_utt += pred.eq(target_te_utt.data.view_as(pred)).sum()
            correct_utt += pred.eq(target_te_utt.data.view_as(pred)).cuda().sum()

            if batch_idx == 0:
                eer_spk_value = output_spk.data.cpu().numpy().copy()
                eer_utt_value = output_utt.data.cpu().numpy().copy()
                eer_spk_label = target_te_spk_numpy.copy()
                eer_utt_label = target_te_utt_numpy.copy()
            else:
                eer_spk_value = np.append(eer_spk_value, output_spk.data.cpu().numpy(), axis=0)
                eer_utt_value = np.append(eer_utt_value, output_utt.data.cpu().numpy(), axis=0)
                eer_spk_label = np.append(eer_spk_label, target_te_spk_numpy, axis=0)
                eer_utt_label = np.append(eer_utt_label, target_te_utt_numpy, axis=0)

            cnt += self.args.batch_size

        correct_utt = correct_utt.cpu().numpy()
        acc_utt = (100 * correct_utt / len(self.data_pack.fea_te_key_list))

        correct_spk = correct_spk.cpu().numpy()
        acc_spk = (100 * correct_spk / len(self.data_pack.fea_te_key_list))

        data_save_path = self.saving_path + '/eer_data_epoch' + str(self.args.epochs)
        np.save((data_save_path + 'spk_value.npy'), eer_spk_value)
        np.save((data_save_path + 'utt_value.npy'), eer_utt_value)
        np.save((data_save_path + 'spk_label.npy'), eer_spk_label)
        np.save((data_save_path + 'utt_label.npy'), eer_utt_label)
        np.save((data_save_path + 'session.npy'), self.data_pack.yte_ses)
        path_model = self.saving_path + '/MODEL_epoch' + str(self.args.epochs) + '.pth'
        np.save((self.saving_path + '/epoch' + str(self.args.epochs) + '_acc_spk' + str(acc_spk) + '_acc_utt' + str(acc_utt)), acc_spk, acc_utt)

        torch.save(self.model.state_dict(), path_model)
        print(path_model + ' Model save successfully!')

        print('\nTest set for spk task: Spk Verification Accuracy: {}/{} ({:.2f}%)'.format(correct_spk, len(self.data_pack.fea_te_key_list), acc_spk))
        print('Test set for utt task: Utt Verification Accuracy: {}/{} ({:.2f}%)\n'.format(correct_utt, len(self.data_pack.fea_te_key_list), acc_utt))
        print('########################################################################################\n')


