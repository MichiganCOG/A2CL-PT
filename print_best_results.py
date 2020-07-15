import os
import numpy as np

list_output = [d for d in os.listdir('./output')]
list_output.sort()

list_aprint = []
for d in list_output:
    list_val = [v for v in os.listdir(os.path.join('./output', d)) if 'val_' in v]
    list_val.sort()
    list_ascore = []
    for v in list_val:
        f = os.path.join('./output', d, v)
        if os.path.isfile(f):
            scores = open(f, 'r').readlines()
            if len(scores) > 1:
                scores = scores[-1].split(': ')
                if len(scores) > 1:
                    scores = scores[1].split(' || ')
                    if len(scores) > 1:
                        list_ascore.append((v, float(scores[0])))

    if list_ascore:
        list_ascore = sorted(list_ascore, key=lambda s: s[1], reverse=True)
        vmax = list_ascore[0][0]
        iters = int(list_val[-1].split('_')[1].split('.')[0])
        list_aprint.append('%s:    %s  @ %d / %d' % (d, open(os.path.join('./output', d, vmax), 'r').readlines()[-1].split(': ')[1], int(vmax.split('_')[1].split('.')[0]), iters))

for p in list_aprint:
    print (p)
