import os
import numpy as np
import torch
import json
from model import Model
from detection import postprocess_filter, detect_segments, postprocess_ambiguous_v
from datetime import timedelta

# example presented in the examples folder (LongJump.mp4)
datapath = './dataset/THUMOS14'
vname = 'video_test_0001369'
idx_end = 360
weight = './weights/best2.pt'

# load data
feature = np.load(os.path.join(datapath, 'feature_val.npy'), allow_pickle=True)[()][vname][:idx_end]
dict_annts = json.load(open(os.path.join(datapath, 'annotation.json'), 'r'))
cnames = dict_annts['list_classes']
set_ambiguous = dict_annts['set_ambiguous']
fps_extracted = dict_annts['miscellany']['fps_extracted']
len_feature_chunk = dict_annts['miscellany']['len_feature_chunk']

# build model
model = Model(len(cnames), 40, 0.6)
model.load_state_dict(torch.load(weight))
model.eval()

# process
tcam = model(torch.from_numpy(np.array([feature])))[-1].data.numpy()
predictions, confidences = postprocess_filter(tcam, 8, 1)
preds_cwise = detect_segments(predictions, confidences, cnames, 0, 0)
preds_cwise = postprocess_ambiguous_v(preds_cwise, set_ambiguous, vname, fps_extracted/len_feature_chunk)

# print in the order of confidence
print ('Detection results of %s:' % vname)
i = 1
for c in preds_cwise:
    for p in preds_cwise[c]:
        s, e = p[1]*len_feature_chunk/fps_extracted, p[2]*len_feature_chunk/fps_extracted
        print ('%3d) %s, %5.1f ~ %5.1f (sec)' % (i, c, s, e))
        i += 1
