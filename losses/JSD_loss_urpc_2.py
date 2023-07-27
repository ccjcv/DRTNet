import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def calc_jsd_multiscale_2(labels1_a, pred1, pred2, pred3,
                          weight_ori, weight_small, weight_large, threshold=0.8):

    Mask_label255 = (labels1_a < 255).float()  # do not compute the area that is irrelavant (dataaug)  b,h,w
    # weight_ori = weight_ori + 1e-8
    # weight_small = weight_small + 1e-8
    # weight_large = weight_large + 1e-8

    criterion1 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion2 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    criterion3 = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    loss1 = criterion1(pred1 * weight_ori + 1e-8, labels1_a)  # * weight_softmax[0]
    loss2 = criterion2(pred2 * weight_small + 1e-8, labels1_a)  # * weight_softmax[1]
    loss3 = criterion3(pred3 * weight_large + 1e-8, labels1_a)  # * weight_softmax[2]
    # loss1 = criterion1(pred1, labels1_a)  # * weight_softmax[0]
    # loss2 = criterion2(pred2, labels1_a)  # * weight_softmax[1]
    # loss3 = criterion3(pred3, labels1_a)  # * weight_softmax[2]

    pred1 = torch.softmax(pred1, dim=1)
    pred2 = torch.softmax(pred2, dim=1)
    pred3 = torch.softmax(pred3, dim=1)

    loss = (loss1 + loss2 + loss3) / 3

    probs = [logits for i, logits in enumerate([pred1, pred2, pred3])]

    # preds = (probs[0] + probs[1] + probs[2]) / 3
    preds = probs[0] * weight_ori + probs[1] * weight_small + probs[2] * weight_large

    # weighted_probs = [weight_softmax[i] * prob for i, prob in enumerate(probs)]  # weight_softmax[i]*
    # mixture_label = (torch.stack(weighted_probs)).sum(axis=0)
    # #mixture_label = torch.clamp(mixture_label, 1e-7, 1)  # h,c,h,w
    # mixture_label = torch.clamp(mixture_label, 1e-3, 1-1e-3)  # h,c,h,w

    # add this code block for early torch version where torch.amax is not available

    max_probs = torch.amax(preds*Mask_label255.unsqueeze(1), dim=(-3, -2, -1), keepdim=True)
    mask = max_probs.ge(threshold).float()


    # logp_mixture = mixture_label.log()

    log_probs = [torch.sum(F.kl_div(torch.log(preds + 1e-8), prob + 1e-8, reduction='none') * mask, dim=1, keepdim=True) for prob in probs]

    # exp_variance_1 = torch.exp(-log_probs[0])
    # exp_variance_2 = torch.exp(-log_probs[1])
    # exp_variance_3 = torch.exp(-log_probs[2])
    # consistency_dist_1 = (preds - probs[0]) ** 2
    # consistency_dist_2 = (preds - probs[1]) ** 2
    # consistency_dist_3 = (preds - probs[2]) ** 2
    # consistency_loss_1 = torch.mean(
    #     consistency_dist_1 * exp_variance_1) / (torch.mean(exp_variance_1) + 1e-8) + torch.mean(log_probs[0])
    # consistency_loss_2 = torch.mean(
    #     consistency_dist_2 * exp_variance_2) / (torch.mean(exp_variance_2) + 1e-8) + torch.mean(log_probs[1])
    # consistency_loss_3 = torch.mean(
    #     consistency_dist_3 * exp_variance_3) / (torch.mean(exp_variance_3) + 1e-8) + torch.mean(log_probs[2])
    # consistency_loss = (consistency_loss_1 + consistency_loss_2 + consistency_loss_3) / 3
    consistency_loss_1 = torch.mean(log_probs[0])
    consistency_loss_2 = torch.mean(log_probs[1])
    consistency_loss_3 = torch.mean(log_probs[2])
    consistency_loss = (consistency_loss_1 + consistency_loss_2 + consistency_loss_3) / 3

    # if Mask_label255_sign == 'yes':
    #     consistency = sum(log_probs)*Mask_label255
    # else:
    #     consistency = sum(log_probs)

    return torch.mean(loss), consistency_loss, preds