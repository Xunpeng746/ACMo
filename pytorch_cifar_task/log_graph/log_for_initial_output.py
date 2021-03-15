import matplotlib

matplotlib.use('Agg')
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


class LOG_REC:
    def __init__(self, win_size):
        self.win_size = win_size
        self.querec = []
        self.quesmo = []

    def queue_addition(self, new_element):
        self.querec.append(new_element)
        head_poi = max(len(self.querec) - self.win_size, 0)
        self.quesmo.append(
            sum([self.querec[idx] for idx in range(head_poi, len(self.querec))]) / max(self.win_size,
                                                                                       len(self.querec) - head_poi))


architecture_list = ['VGGNet_','ResNet_','DenseNet_']

optimizer_dict = {
    'sgd_m_opt': 'b',
    'adam_opt': 'r',
    'amsgrad_opt':'g',
    # 'acutum_wd': 'y',
    'adabound_opt':'gray',
    'adamw_opt': 'c',
    'padam_opt': 'm',
    'radam_opt': 'pink',
    'yogi_opt': 'yellow',
    'acutum_opt': 'orange'

    # 'acutum_OT_opt':'g'
    # 'acutum': 'y',
    # 'acutum_adaptive': 'g',
    # 'vgg_acutum_original_wd_2': 'c',
    # 'vgg_acutum_OT_wd_2': 'g',
    # 'res_acutum_truncated_wd': 'g'
    # 'sgd':'b',
    # 'adam':'r',
    # 'acutum':'y',
    # 'acutum_OT':'g',
    # 'acutum_original_2': 'c'
}


def model_figure(model_now, i, x_len):
    base_root = './cifar10_experimental_results/'
    x_plot = np.arange(0, x_len, 1)
    for mes in optimizer_dict:
        print(mes, i)
        input_file_N = base_root + model_now + mes + '.out'
        if os.path.exists(input_file_N) == False:
            print(input_file_N)
            continue
        print('loading' + input_file_N)
        input_file = open(input_file_N, 'r')
        training_regret_loss = LOG_REC(50)
        test_regret_loss = LOG_REC(50)
        training_acc = LOG_REC(50)
        test_acc = LOG_REC(50)
        while 1:
            lines = input_file.readline()
            if lines == '':
                break
            if lines.find('/50000') != -1:
                loss_head = lines.find('Loss: ') + len('Loss: ')
                loss_end = lines.find('|', loss_head)
                loss_det = float(lines[loss_head:loss_end - 1])
                training_regret_loss.queue_addition(loss_det)

                acc_head = lines.find('Acc: ') + len('Acc: ')
                acc_end = lines.find('%', acc_head)
                acc_det = float(lines[acc_head:acc_end])
                training_acc.queue_addition(acc_det)
            elif lines.find('/10000') != -1:
                loss_head = lines.find('Loss: ') + len('Loss: ')
                loss_end = lines.find('|', loss_head)
                loss_det = float(lines[loss_head:loss_end - 1])
                test_regret_loss.queue_addition(loss_det)

                acc_head = lines.find('Acc: ') + len('Acc: ')
                acc_end = lines.find('%', acc_head)
                acc_det = float(lines[acc_head:acc_end])
                test_acc.queue_addition(acc_det)
            else:
                continue
        #print (test_acc.querec)
        input_file.close()
        total_num = len(architecture_list)
        idx_training = i + 1
        idx_test = i + total_num + 1
        print(idx_training, idx_test)
        # ================== train graph ====================
        plt.subplot(2, total_num, idx_training)
        plt.title("({}) Train Loss and Test Accuracy for {}".format(chr(ord('a')+idx_training-1), model_now.split('_')[0]), fontsize=7)
        # =================== ytick settings ===============
        plt.ylim((0, 0.5))
        train_y_ticks = np.arange(0, 0.6, 0.05)
        # ==================================================
        #plt.ylim((0, 1.8))
        #train_y_ticks = np.arange(0, 1.8, 0.2)
        # =================== ytick settings end ==========
        plt.yticks(train_y_ticks)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.xlabel("Epochs", fontsize=7)
        plt.ylabel("Train Loss", fontsize=7)
        plt.grid(True, linewidth=0.5)
        if mes.split('_')[0]!='acutum':
            plt.plot(x_plot, training_regret_loss.querec[:x_len:1], color=optimizer_dict[mes], linewidth=0.5,
                     label=mes.split('_')[0])
        else:
            plt.plot(x_plot, training_regret_loss.querec[:x_len:1], color=optimizer_dict[mes], linewidth=0.5,
                 label=mes.split('_')[0])
        plt.legend(loc='upper right', fontsize=5)
        # =================== test graph =====================
        plt.subplot(2, total_num, idx_test)
        # =================== ytick settings ===============
        plt.ylim((67, 96))
        test_y_ticks = np.arange(67, 97, 2.5)
        # ==================================================
        # plt.ylim((35, 80))
        # test_y_ticks = np.arange(35, 80, 5)
        # =================== ytick settings end ==========
        plt.yticks(test_y_ticks)
        plt.xlabel("Epochs", fontsize=7)
        plt.ylabel("Test Accuracy %", fontsize=7)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        plt.grid(True, linewidth=0.5)
        if mes.split('_')[0]!='acutum':
            plt.plot(x_plot, test_acc.querec[:x_len:1], color = optimizer_dict[mes], linewidth=0.5,
                     label=mes.split('_')[0])
        else:
            plt.plot(x_plot, test_acc.querec[:x_len:1], color = optimizer_dict[mes], linewidth=0.5,
                     label=mes.split('_')[0])
        plt.legend(loc='lower right', fontsize=5)
        #print(training_regret_loss.querec)


print(range(len(architecture_list)))
plt.figure(figsize=(10,5))
for iter_idx in range(len(architecture_list)):
    model_figure(architecture_list[iter_idx], iter_idx, 200)

pp = PdfPages('result_loss_fin.pdf')
plt.savefig(pp, format='pdf')
pp.close()
