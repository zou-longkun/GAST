import math
import torch

c = torch.tensor([[1,1,1],[4,6,5]])


def entropy(c):
    result = -1
    if len(c) > 0:
        result = 0
    for x in c:
        result += (-x) * math.log(x, 2)
    return result


for i, v in enumerate(c):
    print(i)
    print(entropy(v))


def cross_entroy_loss(x, target, curv_conf, device='cuda:0'):
    rows = x.size(0)
    cols = x.size(1)
    curv_conf = curv_conf.reshape(-1, 1)
    one_hot = torch.zeros((rows, cols)).to(device)
    for i in range(len(target)):
        cols = target[i]
        one_hot[i][cols] = 1
    softmax = torch.exp(x) / torch.sum(torch.exp(x), dim=1).reshape(-1, 1)
    logsoftmax = torch.log(softmax) * curv_conf * 100
    nllloss = - torch.sum(one_hot * logsoftmax) / target.shape[0]
    CrossEntroyLoss_value = nllloss

    return CrossEntroyLoss_value