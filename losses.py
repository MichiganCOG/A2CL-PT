import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd.function import Function

class MILLoss(nn.Module):
    def __init__(self, device, s=8):
        super(MILLoss, self).__init__()
        self.device = device
        self.s = s

    def forward(self, cas, len_features, label):
        # NxT'x101, [T1', T2', ..., TN'], Nx101
        label = label / label.sum(dim=1, keepdim=True) # Nx101
        confidence = torch.zeros(label.shape, device=self.device)
        for i, len_seq in enumerate(len_features):
            if self.s > 0:
                topk = cas[i][:len_seq].topk(k=int(np.ceil(len_seq/self.s)), dim=0)[0] # Ti'x101 -> Ti''x101
            else:
                topk = cas[i][:len_seq]
            confidence[i] = topk.mean(dim=0)

        return -(label*confidence.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

class ACLPT_func(Function):
    @staticmethod
    def forward(ctx, featureH, featureL, label, centers, margins, gamma):
        # N'x1024, (N',), Cx1024
        ctx.save_for_backward(featureH, featureL, label, centers, margins, gamma)
        num_pair = featureH.shape[0]
        num_class = centers.shape[0]
        centers_normed = F.normalize(centers, dim=1) # Cx1024
        featureH_normed = F.normalize(featureH, dim=1) # N'x1024
        distmatH = torch.acos(torch.mm(featureH_normed, centers_normed.t())) # N'xC
        featureL_normed = F.normalize(featureL, dim=1)
        distmatL = torch.acos(torch.mm(featureL_normed, centers_normed.t()))
        mask = distmatH.new_zeros(distmatH.size(), dtype=torch.long)
        mask.scatter_add_(1, label.long().unsqueeze(1), torch.ones(num_pair, 1, device=mask.device, dtype=torch.long))

        distHic = distmatH[mask==1]
        distLic = distmatL[mask==1]
        distHicL = torch.min(distmatH[mask==0].view(num_pair, num_class-1), dim=1)[0]
        li1 = distHic-distLic+margins[0]
        li2 = distHic-distHicL+margins[1]
        loss = li1[li1>0].sum() * gamma[0] + li2[li2>0].sum()
        return loss/num_pair

    @staticmethod
    def backward(ctx, grad_output):
        featureH, featureL, label, centers, margins, gamma = ctx.saved_tensors
        num_pair = featureH.shape[0]
        num_class = centers.shape[0]
        centers_normed = F.normalize(centers, dim=1) # Cx1024
        featureH_normed = F.normalize(featureH, dim=1) # N'x1024
        distmatH = torch.mm(featureH_normed, centers_normed.t()) # N'xC
        featureL_normed = F.normalize(featureL, dim=1)
        distmatL = torch.mm(featureL_normed, centers_normed.t())
        mask = distmatH.new_zeros(distmatH.size(), dtype=torch.long)
        mask.scatter_add_(1, label.long().unsqueeze(1), torch.ones(num_pair, 1, device=mask.device, dtype=torch.long))

        distHic = distmatH[mask==1]
        distLic = distmatL[mask==1]
        distHicL, hard_index_batch = torch.max(distmatH[mask==0].view(num_pair, num_class-1), dim=1)
        hard_index_batch[hard_index_batch>=label] += 1
        li1 = torch.acos(distHic)-torch.acos(distLic)+margins[0]
        li2 = torch.acos(distHic)-torch.acos(distHicL)+margins[1]

        centers_normed_batch = centers_normed.index_select(0, label.long())
        hard_normed_batch = centers_normed.index_select(0, hard_index_batch)

        d = -(1-distHic.pow(2)).pow(-0.5)
        e = -(1-distHicL.pow(2)).pow(-0.5)
        f = -(1-distLic.pow(2)).pow(-0.5)
        I = torch.eye(featureH.shape[1], device=d.device)
        xcH = (I-torch.einsum('bi,bj->bij', (featureH_normed, featureH_normed)))/featureH.norm(dim=1, keepdim=True).unsqueeze(-1)
        xcL = (I-torch.einsum('bi,bj->bij', (featureL_normed, featureL_normed)))/featureL.norm(dim=1, keepdim=True).unsqueeze(-1)
        cc = (I-torch.einsum('bi,bj->bij', (centers_normed, centers_normed)))/centers.norm(dim=1, keepdim=True).unsqueeze(-1)
        d = d.unsqueeze(1)
        e = e.unsqueeze(1)
        f = f.unsqueeze(1)

        counts_h = centers.new_ones(num_class) # (C,)
        counts_hl = centers.new_ones(num_class) # (C,)
        counts_c = centers.new_ones(num_class) # (C,)
        ones_h = centers.new_ones(num_pair) # (N',)
        ones_h[li2<=0] = 0
        ones_c = centers.new_ones(num_pair) # (N',)
        ones_c[li1<=0] = 0
        grad_centers = centers.new_zeros(centers.size()) # Cx1024

        counts_h.scatter_add_(0, label.long(), ones_h)
        counts_hl.scatter_add_(0, hard_index_batch, ones_h)
        counts_c.scatter_add_(0, label.long(), ones_c)

        grad_centers_h = featureH_normed * d
        grad_centers_h[li2<=0] = 0
        grad_centers += torch.scatter_add(centers.new_zeros(centers.size()), 0, label.unsqueeze(1).expand(featureH_normed.size()).long(), grad_centers_h)/counts_h.unsqueeze(-1)

        grad_centers_hl = -featureH_normed * e
        grad_centers_hl[li2<=0] = 0
        grad_centers += torch.scatter_add(centers.new_zeros(centers.size()), 0, hard_index_batch.unsqueeze(1).expand(featureH_normed.size()), grad_centers_hl)/counts_hl.unsqueeze(-1)

        grad_centers_c = featureH_normed*d - featureL_normed*f
        grad_centers_c[li1<=0] = 0
        grad_centers += torch.scatter_add(centers.new_zeros(centers.size()), 0, label.unsqueeze(1).expand(featureH_normed.size()).long(), grad_centers_c*gamma)/counts_c.unsqueeze(-1)

        grad_centers /= num_pair

        grad = centers_normed_batch * d - hard_normed_batch * e
        grad[li2<=0] = 0

        grad_h = centers_normed_batch * d
        grad_h[li1<=0] = 0

        grad_h = grad_output * (grad+grad_h*gamma) / num_pair

        grad_l = -centers_normed_batch * f
        grad_l[li1<=0] = 0
        grad_l = grad_output * grad_l*gamma / num_pair

        return torch.bmm(xcH, grad_h.unsqueeze(-1)).squeeze(-1), torch.bmm(xcL, grad_l.unsqueeze(-1)).squeeze(-1), None, torch.bmm(cc, grad_centers.unsqueeze(-1)).squeeze(-1), None, None

class A2CLPTLoss(nn.Module):
    def __init__(self, device, num_class, dim_feature=1024, alpha=1, beta_l=0.001, beta_h=0.1, margin1=2, margin2=1, gamma=0.6):
        super(A2CLPTLoss, self).__init__()
        self.device = device
        self.num_class = num_class
        self.dim_feature = dim_feature
        self.alpha = alpha
        self.beta_l = beta_l
        self.beta_h = beta_h
        self.margin1 = margin1
        self.margin2 = margin2
        self.gamma = gamma
        self.centers1 = nn.Parameter(torch.randn(num_class, dim_feature, device=device))
        self.centers2 = nn.Parameter(torch.randn(num_class, dim_feature, device=device))
        self.normalize_centers()
        self.ith = 5000

    def get_alpha(self):
        return self.alpha

    def normalize_centers(self):
        with torch.no_grad():
            self.centers1.div_(self.centers1.norm(dim=1, keepdim=True))
            self.centers2.div_(self.centers2.norm(dim=1, keepdim=True))

    def forward(self, logits, cas, len_features, label, iters):
        # NxT'x1024, NxT'x101, [T1', T2', ..., TN'], Nx101, scalar
        loss = 0
        list_pair = []
        for j in range(len(len_features)):
            #if label[j].sum() == 1:
            if label[j].sum() == 1 or (label[j].sum() > 0 and iters >= self.ith):
                for c in label[j].nonzero():
                    list_pair.append((j, c.squeeze()))

        if list_pair:
            num_pair = len(list_pair)
            beta1 = torch.FloatTensor(num_pair).uniform_(self.beta_l, self.beta_h).to(self.device)
            beta2 = torch.FloatTensor(num_pair).uniform_(self.beta_l, self.beta_h).to(self.device)
            aHf1 = torch.zeros(num_pair, self.dim_feature, device=self.device) # N'x1024 (N': the number of features with annotation)
            aLf1 = torch.zeros(num_pair, self.dim_feature, device=self.device)
            aHf2 = torch.zeros(num_pair, self.dim_feature, device=self.device) # N'x1024 (N': the number of features with annotation)
            aLf2 = torch.zeros(num_pair, self.dim_feature, device=self.device)
            lab = torch.zeros(num_pair, device=self.device) # (N',)
            for i, (j, c) in enumerate(list_pair):
                atn1 = F.softmax(cas[0][j][:len_features[j]], dim=0) # Ti'x101
                atn1L = F.softmax(beta1[i]*cas[0][j][:len_features[j]], dim=0)
                Hf1 = torch.mm(logits[j][:len_features[j]].permute(1,0), atn1) # 1024xTi', Ti'x101-> 1024x101
                Lf1 = torch.mm(logits[j][:len_features[j]].permute(1,0), atn1L)
                aHf1[i] = Hf1[:,c] # (1024,)
                aLf1[i] = Lf1[:,c]
                atn2 = F.softmax(cas[1][j][:len_features[j]], dim=0) # Ti'x101
                atn2L = F.softmax(beta2[i]*cas[1][j][:len_features[j]], dim=0)
                Hf2 = torch.mm(logits[j][:len_features[j]].permute(1,0), atn2) # 1024xTi', Ti'x101-> 1024x101
                Lf2 = torch.mm(logits[j][:len_features[j]].permute(1,0), atn2L)
                aHf2[i] = Hf2[:,c] # (1024,)
                aLf2[i] = Lf2[:,c]
                lab[i] = c

            loss = ACLPT_func.apply(aHf1, aLf1, lab, self.centers1, torch.FloatTensor([self.margin1, self.margin2]).to(self.device), torch.FloatTensor([self.gamma]).to(self.device))
            loss += ACLPT_func.apply(aHf2, aLf2, lab, self.centers2, torch.FloatTensor([self.margin1, self.margin2]).to(self.device), torch.FloatTensor([self.gamma]).to(self.device))
            loss *= self.alpha

        return loss

