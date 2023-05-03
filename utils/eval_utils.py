import itertools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from thop import profile
import time 

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    elif args.model_type == 'patch_gcn':
        from models.model_SGMF import SGMF
        model_dict = {'num_layers': args.num_gcn_layers, 'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = SGMF(**model_dict)
    elif args.model_type == 'dgc':
        from models.model_graph_mil import DeepGraphConv_Surv
        model_dict = {'edge_agg': args.edge_agg, 'resample': args.resample, 'n_classes': args.n_classes}
        model = DeepGraphConv_Surv(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)

    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, test_error, auc, test_f1, df, acc_logger, attention_result = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, test_error, auc, test_f1, df, each_class_acc, attention_result

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.
    flops = 0.
    params = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    all_pred = []
    all_label = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    attention_result = {}
    begin = time.clock()
    for batch_idx, (data_s, data_l, tissue_data, label, attn_mask_s, attn_mask_l) in enumerate(loader):
        data_s, data_l, tissue_data, label = data_s.to(device), data_l.to(device), tissue_data.to(device), label.to(device)
        attn_mask_s, attn_mask_l = attn_mask_s.to(device), attn_mask_l.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            logits, Y_prob, Y_hat, A_fusion= model(data_s, data_l, tissue_data, attn_mask_s, attn_mask_l)
            # each_flops, each_params = profile(model, inputs=(data_s, data_l, tissue_data, attn_mask_s, attn_mask_l, ))

        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})

        # save tissue attention value to folder
        attention_result[slide_id] = A_fusion

        error = calculate_error(Y_hat, label)
        test_error += error
        # flops += each_flops
        # params += each_params

        all_pred.append(Y_hat.cpu())
        all_label.append(label.cpu())

    test_error /= len(loader)
    test_f1 = f1_score(all_label, all_pred, average='macro')

    # flops /= len(loader)
    # params /= len(loader)
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')

    end = time.clock()
    run_time = (end-begin) / len(loader) 
    print('run time = '+ str(run_time))

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, test_f1, df, acc_logger, attention_result

def plot_confusion_matrix(cm, classes, save_path,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    绘制混淆矩阵
    """
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()
