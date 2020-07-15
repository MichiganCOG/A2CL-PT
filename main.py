import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import random
from dataset import wstalDataset
from model import Model
from losses import MILLoss, A2CLPTLoss
from classification import calc_classification_mAP
from detection import calc_detection_mAP
import time
from datetime import timedelta

parser = argparse.ArgumentParser(description='wstal')
parser.add_argument('--mode', help='train | val')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataset', default='THUMOS14', help='THUMOS14 | ActivityNet1.2 | ActivityNet1.3')
parser.add_argument('--datapath', default='./dataset', help='path to dataset')
parser.add_argument('--weight', default='./weights/best2.pt', help='path to the weight file')
parser.add_argument('--valjump', type=int, default=35000, help='evaluation process starts after [valjump] iterations')
parser.add_argument('--valstep', type=int, default=250, help='evaluation process is performed for every [valstep] iterations')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--b', type=int, default=32, help='batch size of dataloader')
parser.add_argument('--iters', type=int, default=45000, help='number of iterations for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')

parser.add_argument('--maxlen', type=int, default=200, help='max snippet length during training')
parser.add_argument('--margin1', type=float, default=2, help='margin1')
parser.add_argument('--margin2', type=float, default=1, help='margin2')
parser.add_argument('--alpha_r', type=float, default=1, help='alpha for rgb stream')
parser.add_argument('--alpha_f', type=float, default=1, help='alpha for flow stream')
parser.add_argument('--lr_cent_r', type=float, default=0.1, help='learning rate for centers of rgb stream')
parser.add_argument('--lr_cent_f', type=float, default=0.2, help='learning rate for centers of flow stream')
parser.add_argument('--gamma', type=float, default=0.6, help='for balancing NT loss')
parser.add_argument('--beta_l', type=float, default=0.001, help='lower limit of beta for NT')
parser.add_argument('--beta_h', type=float, default=0.1, help='higher limit of beta for NT')
parser.add_argument('--sa', type=float, default=40, help='hyperparameter for adversarial branch')
parser.add_argument('--s', type=float, default=8, help='hyperparameter for classification loss')
parser.add_argument('--omega', type=float, default=0.6, help='relative importance of adversarial branch')


def main():
    global args, device
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_name = '%s_%d_%s_%d' % (args.dataset, args.maxlen, time.strftime("%m-%d_%H-%M-%S"), args.seed)
    if 'val' in args.mode:
        if os.path.isfile(args.weight):
            exp_name = 'val_'+'.'.join(args.weight.split('/')[-1].split('.')[:-1])
        else:
            raise ValueError('unknown weight: '+args.weight)

    args.datapath = os.path.join(args.datapath, args.dataset)
    if 'train' in args.mode:
        train_loader = wstalDataset(args.datapath, args.mode, len_snippet=args.maxlen, batch=args.b)
    val_loader = wstalDataset(args.datapath, 'val', len_snippet=None, batch=1)

    model, dim_feature = load_model(val_loader.get_num_classes())
    if 'val' in args.mode:
        list_assigned_param_name = load_and_assign_weights(model)
    print_args(exp_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if torch.cuda.is_available():
        print ('run on cuda')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        if args.ngpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(args.ngpu))
    else:
        print ('run on cpu')
        device = torch.device("cpu")

    model = model.to(device)
    criteria = set_criteria(val_loader.get_num_classes(), dim_feature)

    if 'train' in args.mode:
        train(train_loader, val_loader, model, criteria, optimizer, exp_name)
        os.system('python print_best_results.py')

    elif 'val' in args.mode:
        val(val_loader, model, exp_name, 0)


def set_criteria(num_class, dim_feature):
    criteria = {}
    criteria['MILLoss'] = MILLoss(device, args.s)

    criteria['ACLPT_r'] = {}
    criteria['ACLPT_r']['criterion'] = A2CLPTLoss(device, num_class, dim_feature=dim_feature, alpha=args.alpha_r, beta_l=args.beta_l, beta_h=args.beta_h, margin1=args.margin1, margin2=args.margin2, gamma=args.gamma)
    criteria['ACLPT_r']['optimizer'] = torch.optim.SGD(criteria['ACLPT_r']['criterion'].parameters(), lr=args.lr_cent_r)
    criteria['ACLPT_f'] = {}
    criteria['ACLPT_f']['criterion'] = A2CLPTLoss(device, num_class, dim_feature=dim_feature, alpha=args.alpha_f, beta_l=args.beta_l, beta_h=args.beta_h, margin1=args.margin1, margin2=args.margin2, gamma=args.gamma)
    criteria['ACLPT_f']['optimizer'] = torch.optim.SGD(criteria['ACLPT_f']['criterion'].parameters(), lr=args.lr_cent_f)

    return criteria


def multiple_loss(criteria, logits_r, cas_r, logits_f, cas_f, tcam, len_features, label, i):
    loss = 0
    loss_mil = criteria['MILLoss'](tcam, len_features, label)
    loss_mil_r = criteria['MILLoss'](cas_r[0], len_features, label) + criteria['MILLoss'](cas_r[1], len_features, label)
    loss_mil_f = criteria['MILLoss'](cas_f[0], len_features, label) + criteria['MILLoss'](cas_f[1], len_features, label)

    loss += (loss_mil+loss_mil_r+loss_mil_f)

    loss_other_r = criteria['ACLPT_r']['criterion'](logits_r, cas_r, len_features, label, i)
    loss_other_f = criteria['ACLPT_f']['criterion'](logits_f, cas_f, len_features, label, i)
    loss += (loss_other_r+loss_other_f)
    criteria['ACLPT_r']['optimizer'].zero_grad()
    criteria['ACLPT_f']['optimizer'].zero_grad()

    return loss


def print_args(exp_name):
    print ('exp_name:      ', exp_name)
    print ('mode:          ', args.mode)
    print ('ngpu:          ', args.ngpu)
    if 'train' in args.mode:
        print ('batch:         ', args.b)
        print ('iters:         ', args.iters)
        print ('lr:            ', args.lr)
        print ('wd:            ', args.wd)
        print ('alpha:         ', args.alpha_r, args.alpha_f)
        print ('lr_cent:       ', args.lr_cent_r, args.lr_cent_f)
        print ('beta:          ', args.beta_l, args.beta_h)
        print ('margins:       ', args.margin1, args.margin2)
        print ('s, sa, omega:  ', args.s, args.sa, args.omega)
        print ('gamma:         ', args.gamma, flush=True)


def val(val_loader, model, exp_name, iters, dmap_avg_max=100):
    if iters:
        path_output = os.path.join('./output', exp_name)
        if not os.path.isdir(path_output):
            os.makedirs(path_output)
        f = open(os.path.join(path_output, 'val_%06d.txt'%iters), 'w')
    else:
        f = None

    model.eval()

    list_labels = []
    list_cas = []
    list_pmf = []

    val_loader.reset_start()
    num_videos = val_loader.get_num_videos()
    for i in range(num_videos):
        data, label = val_loader.load_data_val() # 1xTx2048, (20,)
        list_labels.append(label)

        with torch.no_grad():
            tcam = model(torch.from_numpy(data).to(device)) # 1xTx2048 -> 1xTx20
            cas = tcam[-1][0] # Tx20
            if args.s > 0:
                pmf = torch.mean(torch.topk(cas, k=int(np.ceil(len(cas)/args.s)), dim=0)[0], dim=0) # [20]
            else:
                pmf = torch.mean(cas, dim=0)

        list_pmf.append(F.softmax(pmf, dim=0).cpu().data.numpy())
        list_cas.append(cas.cpu().data.numpy())

    cmap = calc_classification_mAP(np.array(list_pmf), np.array(list_labels))
    list_iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if 'ActivityNet' in args.dataset:
        list_iou = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    list_dmap = calc_detection_mAP(list_cas, list_iou, val_loader, args.s)

    dmap_avg = np.average(list_dmap)

    string_print = ' '*39
    for iou in list_iou:
        string_print += '%5.2f  ' % iou
    string_print = string_print[:-2] + '\n'

    string_print += 'val result at %6d: %5.2f || %5.2f | '%(iters, dmap_avg, cmap)
    for dmap in list_dmap:
        string_print += '%5.2f, ' % dmap
    string_print = string_print[:-2]

    print (string_print, flush=True)

    if f:
        f.write(string_print)
        f.close()

    return dmap_avg


def train(train_loader, val_loader, model, criteria, optimizer, exp_name):
    path_output = os.path.join('./output', exp_name)
    if not os.path.isdir(path_output):
        os.makedirs(path_output)

    dmap_avg_max = 0

    model.train()

    start_time = time.time()
    for i in range(1, args.iters+1):
        data, label, len_features = train_loader.load_data_train() # NxT'x2048, Nx20, [T1', T2', ..., T20']
        data, label = torch.from_numpy(data).to(device), torch.from_numpy(label).to(device)

        logits_r, cas_r, logits_f, cas_f, tcam = model(data)
        loss = multiple_loss(criteria, logits_r, cas_r, logits_f, cas_f, tcam, len_features, label, i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for lkey in [k for k in criteria if 'MIL' not in k]:
            criteria[lkey]['criterion'].normalize_centers()
            criteria[lkey]['optimizer'].step()


        print ('iter: [%6d/%6d] | loss %.4f, %s' % (i, args.iters, loss.item(), timedelta(seconds=int(time.time()-start_time))), flush=True)

        if (i % args.valstep == 0) and (i > args.valjump):
            dmap_avg = val(val_loader, model, exp_name, i, dmap_avg_max)
            if dmap_avg > dmap_avg_max:
                dmap_avg_max = dmap_avg
                torch.save(model.state_dict(), os.path.join(path_output, exp_name+'_%06d.pt' % i))
            model.train()


def load_model(num_class):
    dim_feature = 1024
    model = Model(num_class, args.sa, args.omega)

    return model, dim_feature


def load_and_assign_weights(model):
    if os.path.isfile(args.weight):
        print ('loading weight file: %s' % args.weight)
        weight_dict = torch.load(args.weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])

            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print (' size? ' + name, param.size(), model_dict[name].size())
            else:
                print (' name? ' + name)
    else:
        print ('no weight file. start from scratch')


if __name__ == '__main__':
    main()
