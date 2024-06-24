import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj
from einops import rearrange
from ogb.utils.features import get_bond_feature_dims 


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_CL(X, Y, args):
    if args.CL_similarity_metric == 'InfoNCE_dot_prod':
        criterion = nn.CrossEntropyLoss()
        B = X.size()[0]
        logits = torch.mm(X, Y.transpose(1, 0))  # B*B
        logits = torch.div(logits, args.T)
        labels = torch.arange(B).long().to(logits.device)  # B*1

        CL_loss = criterion(logits, labels)
        pred = logits.argmax(dim=1, keepdim=False)
        CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    elif args.CL_similarity_metric == 'EBM_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
                           for i in range(args.CL_neg_samples)], dim=0)
        neg_X = X.repeat((args.CL_neg_samples, 1))

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(neg_X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + args.CL_neg_samples * loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    elif args.CL_similarity_metric == 'EBM_node_dot_prod':
        criterion = nn.BCEWithLogitsLoss()
        
        neg_index = torch.randperm(len(Y))
        neg_Y = torch.cat([Y[neg_index]])

        pred_pos = torch.sum(X * Y, dim=1) / args.T
        pred_neg = torch.sum(X * neg_Y, dim=1) / args.T

        loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
        loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
        CL_loss = loss_pos + loss_neg

        CL_acc = (torch.sum(pred_pos > 0).float() +
                  torch.sum(pred_neg < 0).float()) / \
                 (len(pred_pos) + len(pred_neg))
        CL_acc = CL_acc.detach().cpu().item()

    else:
        raise Exception

    return CL_loss, CL_acc


def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2


def do_2dTo3d_loss(X, batch, pair_mask, args):
    positions = batch.positions
    positions, _ = to_dense_batch(positions, batch.atom_batch)
    x = (X + rearrange(X, 'b i j c -> b j i c')) * 0.5
    breaks = torch.linspace(args.first_break, args.last_break, steps=args.num_bins-1).to(positions.device)
    sq_breaks = torch.square(breaks)

    dist2 = torch.sum(
        torch.square(
            rearrange(positions,'b l c -> b l () c') - 
            rearrange(positions, 'b l c -> b () l c')),
            dim = -1,
            keepdims = True)
    true_bins = torch.sum(dist2 > sq_breaks, dim=-1)
    errors = -torch.sum(F.one_hot(true_bins, x.shape[-1]) * F.log_softmax(x, dim=-1), dim=-1).unsqueeze(-1)
    avg_error = (torch.sum(errors * pair_mask) / (1e-6 + torch.sum(pair_mask)))

    return avg_error #, torch.sqrt(dist2+16-6)


def do_3dTo2d_loss(X, data, args):
    y = (X + rearrange(X, 'b i j c -> b j i c')) * 0.5
    edge_padding_idx = get_bond_feature_dims()[0]
    edge_type_num = get_bond_feature_dims()[0] + 1
    y = y.view(-1, edge_type_num)

    edge_mask = to_dense_adj(data.edge_index, data.atom_batch)
    edge_type = to_dense_adj(data.edge_index, data.atom_batch, data.edge_attr[..., 0])
    edge_is_mask = torch.full_like(edge_mask, edge_padding_idx)
    gt = torch.where(edge_mask == 1, edge_type, edge_is_mask)
    gt = gt.view(-1)

    error = F.cross_entropy(y, gt.long(), ignore_index=6)
    return error #, torch.sqrt(dist2+16-6)


def get_num_task(dataset):
    """ used in molecule_finetune.py """
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp', 'donor', 'esol', 'freesolv', 'lipophilicity']:
        return 1
    elif dataset == 'pcba':
        return 92
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')