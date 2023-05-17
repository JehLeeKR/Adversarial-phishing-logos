import os
from PIL import Image
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt


vit_extra_tp = [0.8944827586206896, 0.89954]
vit_extra_fp = [0.00098, 0.00098]
vit_extra_iden = [0.8944827586206896, 0.89954]
vit_extra_thres = [0.977, 0.975]
    
swin_extra_tp = [0.83931]
swin_extra_fp = [0.00098]
swin_extra_iden = [0.83931]
swin_extra_thres = [0.98]
    
siamese_extra_tp = [0.94046, 0.93724, 0.93724]
siamese_extra_fp = [0.00098, 0.00049, 0.00049]
siamese_extra_iden = [0.95885, 0.95563, 0.95563]
siamese_extra_thres = [0.817, 0.823, 0.825]
    
siamese_plus_extra_tp = [0.94092, 0.93770, 0.93701]
siamese_plus_extra_fp = [0.00098, 0.00049, 0.00049]
siamese_plus_extra_iden = [0.95885, 0.95563, 0.95563]
siamese_plus_extra_thres = [0.817, 0.823, 0.825]

# Read list to memory
def read_list(file_path):
    # for reading also binary mode is important
    with open(file_path, 'rb') as fp:
        data = pickle.load(fp)
        print('Done loading list')
        return data
    

def plot_tp_threshold(tp_rate_siamese_plus, tp_rate_siamese, tp_rate_vit, tp_rate_swin, thresholds):
    plt.plot(thresholds, tp_rate_vit, color='red', linewidth = 1.5,
         marker='D', markerfacecolor='red', markersize=4, label='ViT')
    plt.plot(thresholds, tp_rate_swin, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(thresholds, tp_rate_siamese, color='blue', linewidth = 1.5, linestyle = 'dashdot',
         marker='v', markerfacecolor='blue', markersize=4, label='Siamese')
    plt.plot(thresholds, tp_rate_siamese_plus, color='magenta', linewidth = 1.5, linestyle = 'dotted',
         marker='x', markerfacecolor='magenta', markersize=4, label='Siamese++')
    plt.xlabel('Threshold Value')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.ylim(top=1.0, bottom=0.5)
    plt.grid()
    plt.savefig('plots/tp_threshold.png')
    plt.show()
    plt.clf()
    return

def plot_fp_threshold(fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin, thresholds):
    plt.plot(sorted(thresholds + vit_extra_thres), sorted(fp_rate_vit + vit_extra_fp, reverse=True), color='red', linewidth = 1.5,
         marker='D', markerfacecolor='red', markersize=4, label='ViT')
    plt.plot(sorted(thresholds + swin_extra_thres), sorted(fp_rate_swin + swin_extra_fp, reverse=True), color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(sorted(thresholds + siamese_extra_thres), sorted(fp_rate_siamese + siamese_extra_fp, reverse=True), color='blue', linewidth = 1.5, linestyle = 'dashdot',
         marker='v', markerfacecolor='blue', markersize=4, label='Siamese')
    plt.plot(sorted(thresholds + siamese_plus_extra_thres), sorted(fp_rate_siamese_plus + siamese_plus_extra_fp, reverse=True), color='magenta', linewidth = 1.5, linestyle = 'dotted',
         marker='x', markerfacecolor='magenta', markersize=4, label='Siamese++')
    plt.xlabel('Threshold Value')
    plt.ylabel('False Positive Rate')
    plt.legend(loc='best')
    plt.yscale('log', nonpositive='clip')
    plt.ylim(top=1.1, bottom=0.5 * 1e-4)
    k = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    labels = ['0', r'$10^{-3}$',r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
    plt.yticks(k, labels)
    plt.grid()
    plt.savefig('plots/fp_threshold.png')
    plt.show()
    plt.clf()
    return

def plot_iden_threshold(iden_rate_siamese_plus, iden_rate_siamese, iden_rate_vit, iden_rate_swin, thresholds):
    plt.plot(thresholds, iden_rate_vit, color='red', linewidth = 1.5,
         marker='D', markerfacecolor='red', markersize=4, label='ViT')
    plt.plot(thresholds, iden_rate_swin, color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(thresholds, iden_rate_siamese, color='blue', linewidth = 1.5, linestyle = 'dashdot',
         marker='v', markerfacecolor='blue', markersize=4, label='Siamese')
    plt.plot(thresholds, iden_rate_siamese_plus, color='magenta', linewidth = 1.5, linestyle = 'dotted',
         marker='x', markerfacecolor='magenta', markersize=4, label='Siamese++')
    plt.xlabel('Threshold Value')
    plt.ylabel('Identification Success Rate')
    plt.legend(loc='best')
    plt.ylim(top=1.0, bottom=0.5)
    plt.grid()
    plt.savefig('plots/iden_threshold.png')
    plt.show()
    plt.clf()
    return

def plot_tp_fp(tp_rate_siamese_plus, tp_rate_siamese, tp_rate_vit, tp_rate_swin, fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin):
    plt.plot(sorted(fp_rate_vit + vit_extra_fp), sorted(tp_rate_vit + vit_extra_tp), color='red', linewidth = 1.5,
         marker='D', markerfacecolor='red', markersize=4, label='ViT')
    plt.plot(sorted(fp_rate_swin + swin_extra_fp), sorted(tp_rate_swin + swin_extra_tp), color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(sorted(fp_rate_siamese[:17] + siamese_extra_fp), sorted(tp_rate_siamese[:17] + siamese_extra_tp), color='blue', linewidth = 1.5, linestyle = 'dashdot',
         marker='v', markerfacecolor='blue', markersize=4, label='Siamese')
    plt.plot(sorted(fp_rate_siamese_plus[:17] + siamese_plus_extra_fp), sorted(tp_rate_siamese_plus[:17] + siamese_plus_extra_tp), color='magenta', linewidth = 1.5, linestyle = 'dotted',
         marker='x', markerfacecolor='magenta', markersize=4, label='Siamese++')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    k = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    # labels = ['0', r'$10^{-3}$',r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
    # plt.xticks(k, labels)
    plt.xlim(left=0.5 * 1e-4, right=1.1)
    plt.legend(loc='best')
    # plt.xscale('log')
    plt.ylim(top=1.0, bottom=0.5)
    plt.grid()
    plt.savefig('plots/tp_fp.png')
    plt.show()
    plt.clf()
    return


def plot_iden_fp(iden_rate_siamese_plus, iden_rate_siamese, iden_rate_vit, iden_rate_swin, fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin):
    plt.plot(sorted(fp_rate_vit + vit_extra_fp), sorted(iden_rate_vit + vit_extra_iden), color='red', linewidth = 1.5,
         marker='D', markerfacecolor='red', markersize=4, label='ViT')
    plt.plot(sorted(fp_rate_swin + swin_extra_fp) , sorted(iden_rate_swin + swin_extra_iden), color='green', linewidth = 1.5, linestyle = 'dashed',
         marker='s', markerfacecolor='green', markersize=4, label='Swin')
    plt.plot(sorted(fp_rate_siamese[:17] + siamese_extra_fp), sorted(iden_rate_siamese[:17] + siamese_extra_iden), color='blue', linewidth = 1.5, linestyle = 'dashdot',
         marker='v', markerfacecolor='blue', markersize=4, label='Siamese')
    plt.plot(sorted(fp_rate_siamese_plus[:17] + siamese_plus_extra_fp), sorted(iden_rate_siamese_plus[:17] + siamese_plus_extra_iden), color='magenta', linewidth = 1.5, linestyle = 'dotted',
         marker='x', markerfacecolor='magenta', markersize=4, label='Siamese++')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Classification Rate')
    plt.legend(loc='best')
    plt.xscale('log', nonpositive='clip')
    k = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    labels = [0, r'$10^{-3}$',r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$']
    plt.xticks(k, labels)
    plt.xlim(left=0.5 * 1e-4, right=1.1)
    plt.ylim(top=1.0, bottom=0.5)
    plt.grid()
    plt.savefig('plots/iden_fp.png')
    plt.show()
    plt.clf()
    return

if __name__ == "__main__":
    # import pandas as pd
    # # vit
    # tp_rate_vit = read_list('metrics_out/tp_fp/tp_rates_vit.txt')
    # fp_rate_vit = read_list('metrics_out/tp_fp/fp_rates_vit.txt')
    
    # # swin
    # tp_rate_swin = read_list('metrics_out/tp_fp/tp_rates_swin.txt')
    # fp_rate_swin = read_list('metrics_out/tp_fp/fp_rates_swin.txt')

    # # siamese
    # tp_rate_siamese = read_list('classification/phishpedia_data/tp_rates_224.txt')
    # fp_rate_siamese = read_list('classification/phishpedia_data/fp_rates_224.txt')

    # # siamese_plus
    # tp_rate_siamese_plus = read_list('classification/phishpedia_data/step_relu/tp_rates_224.txt')
    # fp_rate_siamese_plus = read_list('classification/phishpedia_data/step_relu/fp_rates_224.txt')

    # df_roc = pd.DataFrame({   'ViT.TPR':tp_rate_vit, 
    #                           'ViT.FPR': fp_rate_vit,                                    
    #                           'Swin.TPR':tp_rate_swin, 
    #                           'Swin.FPR': fp_rate_swin,                                    
    #                           'Siamese.TPR':tp_rate_siamese, 
    #                           'Siamese.FPR': fp_rate_siamese,
    #                           'Siamese++.TPR':tp_rate_siamese_plus,
    #                           'Siamese++.FPR': fp_rate_siamese_plus})
    
    # df_roc.to_csv(f"./plots/csv/Vanilla_ROC.csv")
    # exit()
    
    
    
    # siamese_plus
    tp_rate_siamese_plus = read_list('classification/phishpedia_data/step_relu/tp_rates_224.txt')
    fp_rate_siamese_plus = read_list('classification/phishpedia_data/step_relu/fp_rates_224.txt')
    # for i in range(len(fp_rate_siamese_plus)):
    #     if fp_rate_siamese_plus[i] == 0:
    #         fp_rate_siamese_plus[i] += 1e-4
    iden_rate_siamese_plus = read_list('classification/phishpedia_data/step_relu/iden_rates_224.txt')
    
#     assert len(tp_rate_siamese_plus) == 20
#     assert len(fp_rate_siamese_plus) == 20
#     assert len(iden_rate_siamese_plus) == 20
#     # print(fp_rate_siamese_plus[12])
#     # print(fp_rate_siamese_plus)

    # siamese
    tp_rate_siamese = read_list('classification/phishpedia_data/tp_rates_224.txt')
    fp_rate_siamese = read_list('classification/phishpedia_data/fp_rates_224.txt')
    # for i in range(len(fp_rate_siamese)):
    #     if fp_rate_siamese[i] == 0:
    #         fp_rate_siamese[i] += 1e-4
    iden_rate_siamese = read_list('classification/phishpedia_data/iden_rates_224.txt')
#     assert len(tp_rate_siamese) == 20
#     assert len(fp_rate_siamese) == 20
#     assert len(iden_rate_siamese) == 20
#     # print(fp_rate_siamese[12])
    # print(fp_rate_siamese)
    # print(tp_rate_siamese)
    # print(fp_rate_siamese)
    # print(iden_rate_siamese)
    
    # vit
    tp_rate_vit = read_list('metrics_out/tp_fp/tp_rates_vit.txt')
    fp_rate_vit = read_list('metrics_out/tp_fp/fp_rates_vit.txt')
    # for i in range(len(fp_rate_vit)):
    #     if fp_rate_vit[i] == 0:
    #         fp_rate_vit[i] += 1e-4
    iden_rate_vit = read_list('metrics_out/identification/vit.txt')
    
#     assert len(fp_rate_vit) == 20
#     assert len(tp_rate_vit) == 20
#     assert len(iden_rate_vit) == 20
#     print(iden_rate_vit[17])
#     # print(fp_rate_vit)
    print(tp_rate_vit)
    print(fp_rate_vit)
    print(iden_rate_vit)

    # swin
    tp_rate_swin = read_list('metrics_out/tp_fp/tp_rates_swin.txt')
    fp_rate_swin = read_list('metrics_out/tp_fp/fp_rates_swin.txt')
    iden_rate_swin = read_list('metrics_out/identification/swin.txt')
    
#     assert len(fp_rate_swin) == 20
#     assert len(tp_rate_swin) == 20
     # assert len(iden_rate_swin) == 20
    
    # print(iden_rate_swin[18])
#     # print(fp_rate_swin)
    print(tp_rate_swin)
    print(fp_rate_swin)
    print(iden_rate_swin)
    
    thresholds = []
    starting = 0.61
    ending = 1.0
    stride = 0.02
    threshold = starting
    while threshold < ending:
        thresholds.append(threshold)
        threshold += stride

    print('File loaded')
    
    # plot_tp_threshold(tp_rate_siamese_plus, tp_rate_siamese, tp_rate_vit, tp_rate_swin, thresholds)
    # plot_fp_threshold(fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin, thresholds)
    # plot_iden_threshold(iden_rate_siamese_plus, iden_rate_siamese, iden_rate_vit, iden_rate_swin, thresholds)
    
    # tp_rate_swin_new = tp_rate_swin + [0.60023]
    # fp_rate_swin_new = fp_rate_swin + [0.00098]
    # iden_rate_swin_new = iden_rate_swin + [0.60023]
    # plot_iden_fp(iden_rate_siamese_plus, iden_rate_siamese, iden_rate_vit, iden_rate_swin, fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin)
    
    plot_tp_fp(tp_rate_siamese_plus, tp_rate_siamese, tp_rate_vit, tp_rate_swin, fp_rate_siamese_plus, fp_rate_siamese, fp_rate_vit, fp_rate_swin)
    
    

    # fooling_ratio = read_list('metrics_out/fooling_ratio/siamese_fooling_ratio_final.txt')
    # print(fooling_ratio)

    