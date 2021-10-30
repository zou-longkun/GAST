import numpy as np
import random
import open3d
import torch
import json
import torch.nn as nn
import torch.optim as optim
import math
from emd import earth_mover_distance
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, NUM_POINTS
from Models_Norm import PointNet, DGCNN
from utils import pc_utils_Norm, loss, log
import DefRec
import RotCls
import DefCls
import NormReg
import PCM

NWORKERS = 4
MAX_LOSS = 9 * (10 ** 9)


def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==================
# Argparse
# ==================
parser = argparse.ArgumentParser(description='DA on Point Clouds')
parser.add_argument('--exp_name', type=str, default='GAST', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='../DefRec_and_PCM-master/data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=200, help='number of episode to train')
parser.add_argument('--model', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='Model to use')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                    help='comma delimited of gpu ids to use. Use "-1" for cpu usage')
parser.add_argument('--DefRec_dist', type=str, default='volume_based_voxels', metavar='N',
                    choices=['volume_based_voxels', 'volume_based_radius'],
                    help='distortion of points')
parser.add_argument('--num_regions', type=int, default=3, help='number of regions to split shape by')
parser.add_argument('--DefRec_on_src', type=str2bool, default=False, help='Using DefRec in source')
parser.add_argument('--DefRec_on_trgt', type=str2bool, default=False, help='Using DefRec in target')
parser.add_argument('--DefCls_on_src', type=str2bool, default=True, help='Using DefCls in source')
parser.add_argument('--DefCls_on_trgt', type=str2bool, default=True, help='Using DefCls in target')
parser.add_argument('--RotCls_on_src', type=str2bool, default=False, help='Using RotCls in source')
parser.add_argument('--RotCls_on_trgt', type=str2bool, default=False, help='Using RotCls in target')
parser.add_argument('--NormReg_on_src', type=str2bool, default=False, help='Using NormReg in source')
parser.add_argument('--NormReg_on_trgt', type=str2bool, default=False, help='Using NormReg in target')
parser.add_argument('--Dec_on_src', type=str2bool, default=False, help='Using Decoder in source')
parser.add_argument('--Dec_on_trgt', type=str2bool, default=False, help='Using Decoder in target')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--apply_GRL', type=str2bool, default=False, help='Using gradient reverse layer')
parser.add_argument('--apply_SPL', type=str2bool, default=False, help='Using self-paced learning')
parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--grl_weight', type=float, default=0.5, help='weight of the GRL loss')
parser.add_argument('--spl_weight', type=float, default=0.5, help='weight of the SPL loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--DefCls_weight', type=float, default=0.5, help='weight of the DefCls loss')
parser.add_argument('--RotCls_weight', type=float, default=0.2, help='weight of the RotCls loss')
parser.add_argument('--NormReg_weight', type=float, default=0.5, help='weight of the NormReg loss')
parser.add_argument('--Decoder_weight', type=float, default=2.0, help='weight of the Decoder loss')
parser.add_argument('--output_pts', type=int, default=512, help='number of decoder points')
parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--wd', type=float, default=5e-5, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--gamma', type=float, default=0.1, help='threshold for pseudo label')

args = parser.parse_args()

# ==================
# init
# ==================
io = log.IOStream(args)
io.cprint(str(args))

random.seed(1)
# np.random.seed(1)  # to get the same point choice in ModelNet and ScanNet leave it fixed
torch.manual_seed(args.seed)
args.cuda = (args.gpus[0] >= 0) and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.gpus[0]) if args.cuda else "cpu")
if args.cuda:
    io.cprint('Using GPUs ' + str(args.gpus) + ',' + ' from ' +
              str(torch.cuda.device_count()) + ' devices available')
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
else:
    io.cprint('Using CPU')


# ==================
# Read Data
# ==================
def split_set(dataset, domain, set_type="source"):
    """
    Input:
        dataset
        domain - modelnet/shapenet/scannet
        type_set - source/target
    output:
        train_sampler, valid_sampler
    """
    train_indices = dataset.train_ind
    val_indices = dataset.val_ind
    unique, counts = np.unique(dataset.label[train_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " train part: " + str(dict(zip(unique, counts))))
    unique, counts = np.unique(dataset.label[val_indices], return_counts=True)
    io.cprint("Occurrences count of classes in " + set_type + " " + domain +
              " validation part: " + str(dict(zip(unique, counts))))
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


src_dataset = args.src_dataset
trgt_dataset = args.trgt_dataset
data_func = {'modelnet': ModelNet, 'scannet': ScanNet, 'shapenet': ShapeNet}

src_trainset = data_func[src_dataset](io, args.dataroot, 'train')
trgt_trainset = data_func[trgt_dataset](io, args.dataroot, 'train')
trgt_testset = data_func[trgt_dataset](io, args.dataroot, 'test')

# Creating data indices for training and validation splits:
src_train_sampler, src_valid_sampler = split_set(src_trainset, src_dataset, "source")
trgt_train_sampler, trgt_valid_sampler = split_set(trgt_trainset, trgt_dataset, "target")

# dataloaders for source and target
src_train_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                              sampler=src_train_sampler, drop_last=True)
src_val_loader = DataLoader(src_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                            sampler=src_valid_sampler)
trgt_train_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.batch_size,
                               sampler=trgt_train_sampler, drop_last=True)
trgt_val_loader = DataLoader(trgt_trainset, num_workers=NWORKERS, batch_size=args.test_batch_size,
                             sampler=trgt_valid_sampler)
trgt_test_loader = DataLoader(trgt_testset, num_workers=NWORKERS, batch_size=args.test_batch_size)

# ==================
# Init Model
# ==================
if args.model == 'pointnet':
    model = PointNet(args)
elif args.model == 'dgcnn':
    model = DGCNN(args)
else:
    raise Exception("Not implemented")

model = model.to(device)

# Handle multi-gpu
if (device.type == 'cuda') and len(args.gpus) > 1:
    model = nn.DataParallel(model, args.gpus)
best_model = copy.deepcopy(model)

# ==================
# Optimizer
# ==================
opt = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd) if args.optimizer == "SGD" \
    else optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = CosineAnnealingLR(opt, args.epochs - 10)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
criterion_ls = loss.LabelSmoothingCrossEntropy()
criterion_elem = nn.CrossEntropyLoss(reduction='none')  # return the each sample CE over the batch
# lookup table of regions means
lookup = torch.Tensor(pc_utils_Norm.region_mean(args.num_regions)).to(device)


# ==================
# Validation/test
# ==================
def test(test_loader, model=None, set_type="Target", partition="Val", epoch=0):
    # Run on cpu or gpu
    count = 0.0
    print_losses = {'cls': 0.0}
    batch_idx = 0

    with torch.no_grad():
        model.eval()
        test_pred = []
        test_true = []
        for data, labels, _ in test_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]

            logits = model(data, activate_DefRec=False)
            loss = criterion(logits["cls"], labels)
            print_losses['cls'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["cls"].max(dim=1)[1]
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            count += batch_size
            batch_idx += 1

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    test_acc = io.print_progress(set_type, partition, epoch, print_losses, test_true, test_pred)
    conf_mat = metrics.confusion_matrix(test_true, test_pred, labels=list(label_to_idx.values())).astype(int)

    return test_acc, print_losses['cls'], conf_mat


# ==================
# Utils
# ==================
def generate_trgt_pseudo_label(trgt_data, logits, threshold):
    batch_size = trgt_data.size(0)
    pseudo_label = torch.zeros(batch_size, 10).long()  # one-hot label
    sfm = nn.Softmax(dim=1)
    cls_conf = sfm(logits['cls'])
    mask = torch.max(cls_conf, 1)  # 2 * b
    for i in range(batch_size):
        index = mask[1][i]
        if mask[0][i] > threshold:
            pseudo_label[i][index] = 1

    return pseudo_label


# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
best_model = io.save_model(model)
src_val_acc_list = []
src_val_loss_list = []
trgt_val_acc_list = []
trgt_val_loss_list = []

for epoch in range(args.epochs):
    model.train()
    len_dataloader = min(len(src_train_loader), len(trgt_train_loader))

    # init data structures for saving epoch stats
    cls_type = 'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {'total': 0.0, cls_type: 0.0}
    if args.DefRec_on_src:
        src_print_losses['DefRec'] = 0.0
    if args.RotCls_on_src:
        src_print_losses['RotCls'] = 0.0
    if args.DefCls_on_src:
        src_print_losses['DefCls'] = 0.0
    if args.NormReg_on_src:
        src_print_losses['NormReg'] = 0.0
    if args.Dec_on_src:
        src_print_losses['Decoder'] = 0.0
    trgt_print_losses = {'total': 0.0}
    if args.DefRec_on_trgt:
        trgt_print_losses['DefRec'] = 0.0
    if args.RotCls_on_trgt:
        trgt_print_losses['RotCls'] = 0.0
    if args.DefCls_on_trgt:
        trgt_print_losses['DefCls'] = 0.0
    if args.NormReg_on_trgt:
        trgt_print_losses['NormReg'] = 0.0
    if args.Dec_on_trgt:
        trgt_print_losses['Decoder'] = 0.0
    if args.apply_SPL:
        trgt_print_losses['SPL'] = 0.0

    if args.apply_GRL:
        src_print_losses['GRL'] = trgt_print_losses['GRL'] = 0.0
    src_count = trgt_count = 0.0

    batch_idx = 1
    for data1, data2 in zip(src_train_loader, trgt_train_loader):
        opt.zero_grad()
        #### source data ####
        if data1 is not None:
            src_data, src_label, src_norm_curv = data1[0].to(device), data1[1].to(device).squeeze(), data1[2].to(device)
            # src_data, src_norm_curv = pc_utils_Norm.dropout_points(src_data, src_norm_curv, 50)
            # src_data, src_norm_curv = pc_utils_Norm.remove_region_points(src_data, src_norm_curv, device)
            # change to [batch_size, num_coordinates, num_points]
            src_fea_pc = pc_utils_Norm.extract_feature_points(src_data, src_norm_curv, 512).to(device)  # [B, 512, 3]
            src_data = src_data.permute(0, 2, 1)
            batch_size = src_data.size()[0]
            src_domain_label = torch.zeros(batch_size).long().to(device)
            src_data_orig = src_data.clone()
            device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

            if args.DefRec_on_src:
                src_data = src_data_orig.clone()
                src_data, src_mask = DefRec.deform_input(src_data, lookup, args.DefRec_dist, device)
                src_logits = model(src_data, activate_DefRec=True)
                loss = DefRec.calc_loss(args, src_logits, src_data_orig, src_mask)
                src_print_losses['DefRec'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.DefCls_on_src:
                src_data = src_data_orig.clone()
                src_data, src_def_label, curv_conf = DefCls.defcls_input(src_data, src_norm_curv, lookup, device)
                src_logits = model(src_data, activate_DefRec=False)
                loss = DefCls.calc_loss(args, src_logits, src_def_label, curv_conf, criterion_elem)
                src_print_losses['DefCls'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.RotCls_on_src:
                src_data = src_data_orig.clone()
                src_data, src_pos_vals = RotCls.rotcls_input(src_data, device)
                src_logits = model(src_data, activate_DefRec=False)
                loss = RotCls.calc_loss(args, src_logits, src_pos_vals, criterion)
                src_print_losses['RotCls'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.NormReg_on_src:
                src_data = src_data_orig.clone()
                # src_data, src_norm_label, src_curv_label = NormReg.normreg_input(src_data, device)
                # src_norm_curv = torch.cat((src_norm_label, src_curv_label), 2).float()
                src_logits = model(src_data, activate_DefRec=False)
                loss = NormReg.calc_loss(args, src_logits, src_norm_curv)
                src_print_losses['NormReg'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.Dec_on_src:
                src_data = src_data_orig.clone()
                src_fea_pc = src_fea_pc.clone()
                src_logits = model(src_data, activate_DefRec=False)
                src_decoder_pc = src_logits["decoder"].view(batch_size, args.output_pts, 3)  # [B, 512, 3]
                loss = args.Decoder_weight * earth_mover_distance(src_fea_pc, src_decoder_pc, transpose=False).sum()
                src_print_losses['Decoder'] += loss.item()
                src_print_losses['total'] += loss.item()
                loss.backward()

            if args.apply_GRL:
                p = float(batch_idx + epoch * len_dataloader) / args.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                src_data = src_data_orig.clone()
                src_logits = model(src_data, alpha, activate_DefRec=False)
                loss = args.grl_weight * criterion(src_logits["domain_cls"], src_domain_label)
                src_print_losses['GRL'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.apply_PCM:
                src_data = src_data_orig.clone()
                src_data, mixup_vals = PCM.mix_shapes(args, src_data, src_label)
                src_logits = model(src_data, activate_DefRec=False)
                loss = PCM.calc_loss(args, src_logits, mixup_vals, criterion)
                src_print_losses['mixup'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            else:
                src_data = src_data_orig.clone()
                # predict with undistorted shape
                src_logits = model(src_data, activate_DefRec=False)
                loss = args.cls_weight * criterion(src_logits["cls"], src_label)
                src_print_losses['cls'] += loss.item() * batch_size
                src_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            src_count += batch_size

        #### target data ####
        if data2 is not None:
            trgt_data, trgt_label, trgt_norm_curv = data2[0].to(device), data2[1].to(device).squeeze(), data2[2].to(device)
            # trgt_data, trgt_norm_curv = pc_utils_Norm.dropout_points(trgt_data, trgt_norm_curv, 50)
            trgt_fea_pc = pc_utils_Norm.extract_feature_points(trgt_data, trgt_norm_curv, 512).to(device)
            trgt_data = trgt_data.permute(0, 2, 1)
            batch_size = trgt_data.size()[0]
            trgt_domain_label = torch.ones(batch_size).long().to(device)
            trgt_data_orig = trgt_data.clone()
            device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

            if args.DefRec_on_trgt:
                trgt_data = trgt_data_orig.clone()
                trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device)
                trgt_logits = model(trgt_data, activate_DefRec=True)
                loss = DefRec.calc_loss(args, trgt_logits, trgt_data_orig, trgt_mask)
                trgt_print_losses['DefRec'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.DefCls_on_trgt:
                trgt_data = trgt_data_orig.clone()
                trgt_data, trgt_def_label, curv_conf = DefCls.defcls_input(trgt_data, trgt_norm_curv, lookup, device)
                trgt_logits = model(trgt_data, activate_DefRec=False)
                loss = DefCls.calc_loss(args, trgt_logits, trgt_def_label, curv_conf, criterion_elem)
                trgt_print_losses['DefCls'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.RotCls_on_trgt:
                trgt_data = trgt_data_orig.clone()
                trgt_data, trgt_pos_vals = RotCls.rotcls_input(trgt_data, device)
                trgt_logits = model(trgt_data, activate_DefRec=False)
                loss = RotCls.calc_loss(args, trgt_logits, trgt_pos_vals, criterion)
                trgt_print_losses['RotCls'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.NormReg_on_trgt:
                trgt_data = trgt_data_orig.clone()
                # trgt_data, trgt_norm_label, trgt_curv_label = NormReg.normreg_input(trgt_data, device)
                # trgt_norm_curv = torch.cat((trgt_norm_label, trgt_curv_label), 2).float()
                trgt_logits = model(trgt_data, activate_DefRec=False)
                loss = NormReg.calc_loss(args, trgt_logits, trgt_norm_curv)
                trgt_print_losses['NormReg'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.Dec_on_trgt:
                trgt_data = trgt_data_orig.clone()
                trgt_fea_pc = trgt_fea_pc.clone()
                trgt_logits = model(trgt_data, activate_DefRec=False)
                trgt_decoder_pc = trgt_logits["decoder"].view(batch_size, args.output_pts, 3)  # [B, 512, 3]
                loss = args.Decoder_weight * earth_mover_distance(trgt_fea_pc, trgt_decoder_pc, transpose=False).sum()
                trgt_print_losses['Decoder'] += loss.item()
                trgt_print_losses['total'] += loss.item()
                loss.backward()

            if args.apply_GRL:
                p = float(batch_idx + epoch * len_dataloader) / args.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                src_data = src_data_orig.clone()
                src_logits = model(src_data, alpha, activate_DefRec=False)
                loss = args.grl_weight * criterion(trgt_logits['domain_cls'], trgt_domain_label)
                trgt_print_losses['GRL'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

            if args.apply_SPL:
                if epoch % 10 == 0:
                    lam = 2 / (1 + math.exp(-1 * 10 * epoch / args.epochs)) - 1  # penalty parameter
                threshold = math.exp(-1 * args.gamma)  # Increase as training progresses

                trgt_data = trgt_data_orig.clone()
                trgt_logits = model(trgt_data, activate_DefRec=False)
                trgt_pseudo_label = generate_trgt_pseudo_label(trgt_data, trgt_logits, threshold)
                trgt_pseudo_label = trgt_pseudo_label.to(device)
                loss = lam * (- torch.sum(
                    torch.nn.functional.log_softmax(trgt_logits['cls'], dim=1) * trgt_pseudo_label)) / batch_size
                trgt_print_losses['SPL'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()
            trgt_count += batch_size
        opt.step()
        batch_idx += 1

        scheduler.step()

    # print progress
    src_print_losses = {k: v * 1.0 / src_count for (k, v) in src_print_losses.items()}
    src_acc = io.print_progress("Source", "Trn", epoch, src_print_losses)
    trgt_print_losses = {k: v * 1.0 / trgt_count for (k, v) in trgt_print_losses.items()}
    trgt_acc = io.print_progress("Target", "Trn", epoch, trgt_print_losses)

    # ===================
    # Validation
    # ===================
    src_val_acc, src_val_loss, src_conf_mat = test(src_val_loader, model, "Source", "Val", epoch)
    trgt_val_acc, trgt_val_loss, trgt_conf_mat = test(trgt_val_loader, model, "Target", "Val", epoch)
    src_val_acc_list.append(src_val_acc)
    src_val_loss_list.append(src_val_loss)
    trgt_val_acc_list.append(trgt_val_acc)
    trgt_val_loss_list.append(trgt_val_loss)

    # save model according to best source model (since we don't have target labels)
    if src_val_acc > src_best_val_acc:
        src_best_val_acc = src_val_acc
        src_best_val_loss = src_val_loss
        trgt_best_val_acc = trgt_val_acc
        trgt_best_val_loss = trgt_val_loss
        best_val_epoch = epoch
        best_epoch_conf_mat = trgt_conf_mat
        best_model = io.save_model(model)

    # with open('convergence.json', 'w') as f:
    #    json.dump((src_val_acc_list, src_val_loss_list, trgt_val_acc_list, trgt_val_loss_list), f)

io.cprint("Best model was found at epoch %d, source validation accuracy: %.4f, source validation loss: %.4f,"
          "target validation accuracy: %.4f, target validation loss: %.4f"
          % (best_val_epoch, src_best_val_acc, src_best_val_loss, trgt_best_val_acc, trgt_best_val_loss))
io.cprint("Best validtion model confusion matrix:")
io.cprint('\n' + str(best_epoch_conf_mat))

# ===================
# Test
# ===================
model = best_model
trgt_test_acc, trgt_test_loss, trgt_conf_mat = test(trgt_test_loader, model, "Target", "Test", 0)
io.cprint("target test accuracy: %.4f, target test loss: %.4f" % (trgt_test_acc, trgt_best_val_loss))
io.cprint("Test confusion matrix:")
io.cprint('\n' + str(trgt_conf_mat))
