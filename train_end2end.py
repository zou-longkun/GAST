import numpy as np
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from utils.pc_utils import random_rotate_one_axis
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
import argparse
import copy
import utils.log
from data.dataloader_Norm import ScanNet, ModelNet, ShapeNet, label_to_idx, NUM_POINTS
from Models_Norm import PointNet, DGCNN
from utils import pc_utils_Norm
import DefRec
import RotCls
import DefCls
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
parser.add_argument('--exp_name', type=str, default='DefRec_PCM', help='Name of the experiment')
parser.add_argument('--out_path', type=str, default='./experiments', help='log folder path')
parser.add_argument('--dataroot', type=str, default='./data', metavar='N', help='data path')
parser.add_argument('--src_dataset', type=str, default='shapenet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--trgt_dataset', type=str, default='scannet', choices=['modelnet', 'shapenet', 'scannet'])
parser.add_argument('--epochs', type=int, default=150, help='number of episode to train')
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
parser.add_argument('--PosReg_on_src', type=str2bool, default=True, help='Using PosReg in source')
parser.add_argument('--PosReg_on_trgt', type=str2bool, default=True, help='Using PosReg in target')
parser.add_argument('--apply_PCM', type=str2bool, default=False, help='Using mixup in source')
parser.add_argument('--apply_GRL', type=str2bool, default=False, help='Using gradient reverse layer')
parser.add_argument('--apply_SPL', type=str2bool, default=True, help='Using self-paced learning')
parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size', help='Size of train batch per domain')
parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                    help='Size of test batch per domain')
parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'])
parser.add_argument('--cls_weight', type=float, default=0.5, help='weight of the classification loss')
parser.add_argument('--grl_weight', type=float, default=0.5, help='weight of the GRL loss')
parser.add_argument('--spl_weight', type=float, default=0.5, help='weight of the SPL loss')
parser.add_argument('--DefRec_weight', type=float, default=0.5, help='weight of the DefRec loss')
parser.add_argument('--DefCls_weight', type=float, default=0.5, help='weight of the DefCls loss')
parser.add_argument('--PosReg_weight', type=float, default=0.5, help='weight of the PosReg loss')
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
io = utils.log.IOStream(args)
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
opt_spl = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

scheduler = CosineAnnealingLR(opt, args.epochs)
scheduler_spl = CosineAnnealingLR(opt_spl, args.epochs)
criterion = nn.CrossEntropyLoss()  # return the mean of CE over the batch
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
        for data, labels in test_loader:
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


def select_target_by_conf(trgt_train_loader, model=None):
    pc_list = []
    label_list = []
    sfm = nn.Softmax(dim=1)

    with torch.no_grad():
        model.eval()
        for data in trgt_train_loader:
            data = data[0].to(device)
            data = data.permute(0, 2, 1)

            logits = model(data, activate_DefRec=False)
            cls_conf = sfm(logits['cls'])
            mask = torch.max(cls_conf, 1)  # 2 * b
            index = 0
            for i in mask[0]:
                if i > threshold:
                    pc_list.append(data[index].cpu().numpy())
                    label_list.append(mask[1][index].cpu().numpy())
                index += 1
    return pc_list, label_list


class DataLoad(Dataset):
    def __init__(self, io, data, partition='train'):
        self.partition = partition
        self.pc, self.label = data
        self.num_examples = len(self.pc)

        # split train to train part and validation part
        if partition == "train":
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)

        io.cprint("number of " + partition + " examples in modelnet : " + str(len(self.pc)))
        unique, counts = np.unique(self.label, return_counts=True)
        io.cprint("Occurrences count of classes in modelnet " + partition + " set: " + str(dict(zip(unique, counts))))

    def __getitem__(self, item):
        pointcloud = np.copy(self.pc[item])
        pointcloud = random_rotate_one_axis(pointcloud.transpose(1, 0), "z")
        pointcloud = pointcloud.transpose(1, 0)
        label = np.copy(self.label[item])
        return (pointcloud, label)

    def __len__(self):
        return len(self.pc)

# ==================
# Train
# ==================
src_best_val_acc = trgt_best_val_acc = best_val_epoch = 0
src_best_val_loss = trgt_best_val_loss = MAX_LOSS
best_model = io.save_model(model)

for epoch in range(args.epochs):
    model.train()

    # init data structures for saving epoch stats
    cls_type = 'mixup' if args.apply_PCM else 'cls'
    src_print_losses = {'total': 0.0, cls_type: 0.0}
    if args.DefRec_on_src:
        src_print_losses['DefRec'] = 0.0
    if args.PosReg_on_src:
        src_print_losses['PosReg'] = 0.0
    if args.DefCls_on_src:
        src_print_losses['DefCls'] = 0.0
    trgt_print_losses = {'total': 0.0}
    if args.DefRec_on_trgt:
        trgt_print_losses['DefRec'] = 0.0
    if args.PosReg_on_trgt:
        trgt_print_losses['PosReg'] = 0.0
    if args.DefCls_on_trgt:
        trgt_print_losses['DefCls'] = 0.0
    if args.apply_SPL:
        trgt_print_losses['SPL'] = 0.0

    if args.apply_GRL:
        src_print_losses['GRL'] = trgt_print_losses['GRL'] = 0.0
    src_count = trgt_count = 0.0

    if epoch < 100:
        batch_idx = 1
        for data1, data2 in zip(src_train_loader, trgt_train_loader):
            opt.zero_grad()

            #### source data ####
            if data1 is not None:
                src_data, src_label = data1[0].to(device), data1[1].to(device).squeeze()
                # change to [batch_size, num_coordinates, num_points]
                src_data = src_data.permute(0, 2, 1)
                batch_size = src_data.size()[0]
                src_domain_label = torch.zeros(batch_size).long().to(device)
                src_data_orig = src_data.clone()
                device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

                if args.DefRec_on_src:
                    src_data, src_mask = DefRec.deform_input(src_data, lookup, args.DefRec_dist, device)
                    src_logits = model(src_data, activate_DefRec=True)
                    loss = DefRec.calc_loss(args, src_logits, src_data_orig, src_mask)
                    src_print_losses['DefRec'] += loss.item() * batch_size
                    src_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

                if args.DefCls_on_src:
                    src_data, src_def_label = DefCls.defcls_input(src_data, lookup, device)
                    src_logits = model(src_data, activate_DefRec=False)
                    loss = DefCls.calc_loss(args, src_logits, src_def_label, criterion)
                    src_print_losses['DefCls'] += loss.item() * batch_size
                    src_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

                if args.PosReg_on_src:
                    src_data = src_data_orig.clone()
                    src_data, src_pos_label = RotCls.posreg_input(src_data, device)
                    src_logits = model(src_data, activate_DefRec=False)
                    loss = RotCls.calc_loss(args, src_logits, src_pos_label, criterion)
                    src_print_losses['PosReg'] += loss.item() * batch_size
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

                if args.apply_GRL:
                    src_data = src_data_orig.clone()
                    src_logits = model(src_data, activate_DefRec=False)
                    loss = args.grl_weight * criterion(src_logits["domain_cls"], src_domain_label)
                    src_print_losses['GRL'] += loss.item() * batch_size
                    src_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

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
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device).squeeze()
                trgt_data = trgt_data.permute(0, 2, 1)
                batch_size = trgt_data.size()[0]
                trgt_domain_label = torch.ones(batch_size).long().to(device)
                trgt_data_orig = trgt_data.clone()
                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

                if args.DefRec_on_trgt:
                    trgt_data, trgt_mask = DefRec.deform_input(trgt_data, lookup, args.DefRec_dist, device)
                    trgt_logits = model(trgt_data, activate_DefRec=True)
                    loss = DefRec.calc_loss(args, trgt_logits, trgt_data_orig, trgt_mask)
                    trgt_print_losses['DefRec'] += loss.item() * batch_size
                    trgt_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

                if args.DefCls_on_trgt:
                    trgt_data, trgt_def_label = DefCls.defcls_input(trgt_data, lookup, device)
                    trgt_logits = model(trgt_data, activate_DefRec=False)
                    loss = DefCls.calc_loss(args, trgt_logits, trgt_def_label, criterion)
                    trgt_print_losses['DefCls'] += loss.item() * batch_size
                    trgt_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

                if args.PosReg_on_trgt:
                    trgt_data = trgt_data_orig.clone()
                    trgt_data, trgt_pos_label = RotCls.posreg_input(trgt_data, device)
                    trgt_logits = model(trgt_data, activate_DefRec=False)
                    loss = RotCls.calc_loss(args, trgt_logits, trgt_pos_label, criterion)
                    trgt_print_losses['PosReg'] += loss.item() * batch_size
                    trgt_print_losses['total'] += loss.item() * batch_size
                    loss.backward()

                if args.apply_GRL:
                    trgt_data = trgt_data_orig.clone()
                    trgt_logits = model(trgt_data, activate_DefRec=False)
                    loss = args.grl_weight * criterion(trgt_logits['domain_cls'], trgt_domain_label)
                    trgt_print_losses['GRL'] += loss.item() * batch_size
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

        # save model according to best source model (since we don't have target labels)
        if src_val_acc > src_best_val_acc:
            src_best_val_acc = src_val_acc
            src_best_val_loss = src_val_loss
            trgt_best_val_acc = trgt_val_acc
            trgt_best_val_loss = trgt_val_loss
            best_val_epoch = epoch
            best_epoch_conf_mat = trgt_conf_mat
            best_model = io.save_model(model)

    elif args.apply_SPL:  # epoch >= 100
        model = copy.deepcopy(best_model)
        batch_idx = 1
        lam = 0
        threshold = math.exp(-1 * args.gamma)
        if epoch % 10 == 0:
            lam = 2 / (1 + math.exp(-1 * epoch / args.epochs)) - 1  # Gradually increase penalty parameter
            threshold += 0.05  # Gradually increase confidence threshold
            trgt_select_data = select_target_by_conf(trgt_train_loader, best_model)
            trgt_new_data = DataLoad(io, trgt_select_data)
            trgt_new_train_loader = DataLoader(trgt_new_data, num_workers=NWORKERS, batch_size=args.batch_size, drop_last=True)

        for data1, data2 in zip(src_train_loader, trgt_new_train_loader):
            opt_spl.zero_grad()

            #### source data ####
            if data1 is not None:
                src_data, src_label = data1[0].to(device), data1[1].to(device)
                # change to [batch_size, num_coordinates, num_points]
                src_data = src_data.permute(0, 2, 1)
                batch_size = src_data.size()[0]
                src_data_orig = src_data.clone()
                device = torch.device("cuda:" + str(src_data.get_device()) if args.cuda else "cpu")

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
                trgt_data, trgt_label = data2[0].to(device), data2[1].to(device)
                batch_size = trgt_data.size()[0]
                trgt_data_orig = trgt_data.clone()
                device = torch.device("cuda:" + str(trgt_data.get_device()) if args.cuda else "cpu")

                trgt_data = trgt_data_orig.clone()
                trgt_logits = model(trgt_data, activate_DefRec=False)
                loss = lam * criterion(trgt_logits["cls"], trgt_label)
                trgt_print_losses['SPL'] += loss.item() * batch_size
                trgt_print_losses['total'] += loss.item() * batch_size
                loss.backward()

                trgt_count += batch_size
            opt_spl.step()
            batch_idx += 1

        scheduler_spl.step()

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

        # save model according to best source model (since we don't have target labels)
        if src_val_acc > src_best_val_acc:
            src_best_val_acc = src_val_acc
            src_best_val_loss = src_val_loss
            trgt_best_val_acc = trgt_val_acc
            trgt_best_val_loss = trgt_val_loss
            best_val_epoch = epoch
            best_epoch_conf_mat = trgt_conf_mat
            best_model = io.save_model(model)


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
