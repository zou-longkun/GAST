import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


class OrthogonalMatrixLoss(nn.Module):
    def __init__(self):
        super(OrthogonalMatrixLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        m = torch.bmm(x, x.transpose(1, 2))
        d = m.size()[1]
        diag_sum = 0
        for i in range(batch_size):
            for j in range(d):
                diag_sum += m[i][j][j]
        return (m.sum() - diag_sum) / batch_size


# barlow twins
class OrthogonalMatrixLoss_BT(nn.Module):
    def __init__(self, lamb=0.1):
        super(OrthogonalMatrixLoss_BT, self).__init__()
        self.lamb = lamb

    def forward(self, x):
        batch_size = x.size()[0]
        m = torch.bmm(x, x.transpose(1, 2))
        m_square = m.pow(2)
        d = m.size()[1]
        diag_sum = 0
        off_diag_sum = m_square.sum()
        for i in range(batch_size):
            for j in range(d):
                diag_sum += (1 - 2 * m[i][j][j] + m_square[i][j][j])
                off_diag_sum -= m_square[i][j][j]
        return (diag_sum + self.lamb * off_diag_sum) / batch_size


class BarlowTwins(nn.Module):
    def __init__(self, lamb=0.02):
        super().__init__()
        self.lamb = lamb

    def forward(self, y1, y2):
        # empirical cross-correlation matrix
        c = torch.mm(y1.T, y2)
        c.div_(y1.shape[0])

        # use --scale-loss to multiply the loss by a constant factor
        # see the Issues section of the readme
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lamb * off_diag
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()




