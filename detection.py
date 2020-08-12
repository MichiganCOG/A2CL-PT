# Reference
# https://github.com/sujoyp/wtalc-pytorch/blob/master/detectionMAP.py
# https://github.com/naraysa/3c-net/blob/master/detectionMAP.py

import numpy as np

def postprocess_filter(predictions, hyperparam_s, thresholding):
    predictions_filtered = []
    confidences = []
    for pred in predictions:
        # pred: Tx20 cas vector for i-th video
        ptmp = -pred
        for i in range(ptmp.shape[1]):
            ptmp[:,i].sort()
        ptmp = -ptmp
        if hyperparam_s > 0:
            conf = np.mean(ptmp[:int(np.ceil(ptmp.shape[0]/hyperparam_s)),:], axis=0)
        else:
            conf = np.mean(ptmp, axis=0)
        confidences.append(conf) # conf: (20,) confidence score for each class
        ind = (conf > np.max(conf)/2) * (conf > 0) if thresholding==1 else conf > 0
        predictions_filtered.append(pred * ind)

    return predictions_filtered, confidences

def detect_segments(predictions, confidences, cnames, thresholding, oic):
    preds_cwise = {}
    for c, cn in enumerate(cnames):
        preds = []
        for i in range(len(predictions)):
            cas = predictions[i][:,c] # (T,)
            threshold = np.max(cas) - ( np.max(cas) - np.min(cas) ) / 2 if thresholding==1 else 0
            detected = np.concatenate([np.zeros(1), (cas>threshold).astype('float32'), np.zeros(1)], axis=0)
            detected_diff = [detected[j] - detected[j-1] for j in range(1, len(detected))]
            s = [j for j, d in enumerate(detected_diff) if d==1]
            e = [j for j, d in enumerate(detected_diff) if d==-1]
            for j in range(len(s)):
                l = e[j] - s[j]
                oil = max(l//4, 1)
                score = np.max(cas[s[j]:e[j]]) + confidences[i][c] if oic==0 else np.mean(cas[s[j]:e[j]])-np.mean(cas[max(0,s[j]-oil):min(e[j]+oil, len(cas))])+confidences[i][c]*oic
                if l >= 2:
                    preds.append([i, s[j], e[j], score])

        if preds:
            preds = np.array(preds)

            # Sort the list of predictions for class c based on score
            preds = preds[np.argsort(-preds[:,3])]
            preds_cwise[cn] = preds

    return preds_cwise

def postprocess_ambiguous(preds_cwise, set_ambiguous, vnames, feature_rate):
    if set_ambiguous:
        for cn in preds_cwise:
            preds = preds_cwise[cn]
            num_preds = len(preds)
            ind = np.zeros(num_preds)
            for i in range(num_preds):
                v = vnames[int(preds[i][0])]
                if v in set_ambiguous:
                    for a in set_ambiguous[v]:
                        gt = range(round(a[0]*feature_rate), round(a[1]*feature_rate))
                        pd = range(int(preds[i][1]), int(preds[i][2]))
                        if len(set(gt).intersection(set(pd))) > 0:
                            ind[i] = 1
            preds_cwise[cn] = np.array([preds[i, :] for i in range(num_preds) if ind[i] == 0])

    return preds_cwise

def postprocess_ambiguous_v(preds_cwise, set_ambiguous, v, feature_rate):
    if set_ambiguous and v in set_ambiguous:
        for cn in preds_cwise:
            preds = preds_cwise[cn]
            num_preds = len(preds)
            ind = np.zeros(num_preds)
            for i in range(num_preds):
                for a in set_ambiguous[v]:
                    gt = range(round(a[0]*feature_rate), round(a[1]*feature_rate))
                    pd = range(int(preds[i][1]), int(preds[i][2]))
                    if len(set(gt).intersection(set(pd))) > 0:
                        ind[i] = 1
            preds_cwise[cn] = np.array([preds[i, :] for i in range(num_preds) if ind[i] == 0])

    return preds_cwise

# Inspired by Pascal VOC evaluation tool.
def _ap_from_pr(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])

    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])

    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])

    return ap

def calc_mAP(preds_cwise, annts_cwise, cnames, feature_rate, IoU_threshold):
    ap = []
    for cn in cnames:
        if cn in preds_cwise and cn in annts_cwise:
            preds = preds_cwise[cn] # [i, s[j], e[j], score]
            annts = annts_cwise[cn][:] # [i, a['segment'][0], a['segment'][1]]
            num_annts = len(annts)
            tp, fp = [], []
            for i in range(len(preds)):
                matched = False
                best_iou = 0
                for j in range(len(annts)):
                    if preds[i][0] == annts[j][0]: # if they are of the same video
                        gt = range(round(annts[j][1]*feature_rate), round(annts[j][2]*feature_rate))
                        pd = range(int(preds[i][1]), int(preds[i][2]))
                        IoU = len(set(gt).intersection(set(pd))) / len(set(gt).union(set(pd)))
                        if IoU >= IoU_threshold:
                            matched = True
                            if IoU > best_iou:
                                best_iou = IoU
                                best_j = j
                if matched:
                    del annts[best_j]
                tp.append(float(matched))
                fp.append(1.-float(matched))
            tp_c = np.cumsum(tp)
            fp_c = np.cumsum(fp)
            if sum(tp)==0:
                prc = 0.
            else:
                cur_prec = tp_c / (fp_c+tp_c)
                cur_rec = tp_c / num_annts
                prc = _ap_from_pr(cur_prec, cur_rec)
            ap.append(prc)

    if ap:
        return 100*np.mean(ap)
    else:
        return 0

def calc_detection_mAP(predictions, iou_list, val_loader, hyperparam_s, thresholding_p=1, thresholding_d=0, oic=0):
    cnames = val_loader.get_cnames()
    vnames = val_loader.get_vnames()
    feature_rate = val_loader.get_feature_rate()
    set_ambiguous = val_loader.get_set_ambiguous()
    annts_cwise = val_loader.get_annts_classwise()

    # predictions[i] (list): Tx20 cas vectors for i-th video
    predictions, confidences = postprocess_filter(predictions, hyperparam_s, thresholding_p)

    # preds_cwise[cn] (dict): list of all predicted segments for class name cn
    preds_cwise = detect_segments(predictions, confidences, cnames, thresholding_d, oic)
    preds_cwise = postprocess_ambiguous(preds_cwise, set_ambiguous, vnames, feature_rate)

    dmap_list = []
    for iou in iou_list:
        dmap_list.append(calc_mAP(preds_cwise, annts_cwise, cnames, feature_rate, iou))

    return dmap_list

