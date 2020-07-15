import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from math import ceil, floor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

class Model(nn.Module):
    def __init__(self, num_class, s, omega):
        super(Model, self).__init__()

        self.num_class = num_class
        self.s = s
        self.omega = omega

        D = 1024
        d = 0.7

        self.fc_r = nn.Linear(D, D)
        self.fc1_r = nn.Linear(D, D)
        self.fc_f = nn.Linear(D, D)
        self.fc1_f = nn.Linear(D, D)
        self.classifier_r = nn.Conv1d(D, num_class, kernel_size=1)
        self.classifier_f = nn.Conv1d(D, num_class, kernel_size=1)
        self.classifier_ra = nn.ModuleList([nn.Conv1d(D, 1, kernel_size=1) for i in range(num_class)]) # it can be implemented by conv2d with groups=num_class
        self.classifier_fa = nn.ModuleList([nn.Conv1d(D, 1, kernel_size=1) for i in range(num_class)])

        self.dropout_r = nn.Dropout(d)
        self.dropout_f = nn.Dropout(d)

        self.apply(weights_init)

        self.mul_r = nn.Parameter(data=torch.ones(num_class))
        self.mul_f = nn.Parameter(data=torch.ones(num_class))

    def forward(self, inputs):
        N, T, D = inputs.shape
        D //= 2
        x_r = F.relu(self.fc_r(inputs[:,:,:D]))
        x_f = F.relu(self.fc_f(inputs[:,:,D:]))
        x_r = F.relu(self.fc1_r(x_r)).permute(0,2,1)
        x_f = F.relu(self.fc1_f(x_f)).permute(0,2,1)

        x_r = self.dropout_r(x_r)
        x_f = self.dropout_f(x_f)

        k = max(T-floor(T/self.s), 1)
        cls_x_r = self.classifier_r(x_r).permute(0,2,1)
        cls_x_f = self.classifier_f(x_f).permute(0,2,1)
        cls_x_ra = cls_x_r.new_zeros(cls_x_r.shape)
        cls_x_fa = cls_x_f.new_zeros(cls_x_f.shape)
        cls_x_rat = cls_x_r.new_zeros(cls_x_r.shape)
        cls_x_fat = cls_x_f.new_zeros(cls_x_f.shape)

        mask_value = -100

        for i in range(self.num_class):
            mask_r = cls_x_r[:,:,i]>torch.kthvalue(cls_x_r[:,:,i], k, dim=1, keepdim=True)[0]
            x_r_erased = torch.masked_fill(x_r, mask_r.unsqueeze(1), 0)
            cls_x_ra[:,:,i] = torch.masked_fill(self.classifier_ra[i](x_r_erased).squeeze(1), mask_r, mask_value)
            cls_x_rat[:,:,i] = self.classifier_ra[i](x_r).squeeze(1)

            mask_f = cls_x_f[:,:,i]>torch.kthvalue(cls_x_f[:,:,i], k, dim=1, keepdim=True)[0]
            x_f_erased = torch.masked_fill(x_f, mask_f.unsqueeze(1), 0)
            cls_x_fa[:,:,i] = torch.masked_fill(self.classifier_fa[i](x_f_erased).squeeze(1), mask_f, mask_value)
            cls_x_fat[:,:,i] = self.classifier_fa[i](x_f).squeeze(1)

        tcam = (cls_x_r+cls_x_rat*self.omega) * self.mul_r + (cls_x_f+cls_x_fat*self.omega) * self.mul_f

        return x_r.permute(0,2,1), [cls_x_r, cls_x_ra], x_f.permute(0,2,1), [cls_x_f, cls_x_fa], tcam
