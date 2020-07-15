import os
import torch
import numpy as np
import json

class wstalDataset():
    def __init__(self, datapath, mode, len_snippet, batch):
        self.datapath = datapath
        self.mode = mode
        self.len_snippet = len_snippet
        self.batch = batch

        self.features = np.load(os.path.join(self.datapath, 'feature_%s.npy' % mode), allow_pickle=True)[()]
        self.vnames = sorted(list(self.features.keys()))

        dict_annts = json.load(open(os.path.join(self.datapath, 'annotation.json'), 'r'))

        self.cnames = dict_annts['list_classes']
        self._filter_vnames(dict_annts['database'])
        if 'val' in self.mode:
            self.set_ambiguous = dict_annts['set_ambiguous']
            self.annts_cwise = self._get_annts_cwise(dict_annts['database'])

        self.labels = self._get_labels(dict_annts['database'])
        self.fps_extracted = dict_annts['miscellany']['fps_extracted']
        self.len_feature_chunk = dict_annts['miscellany']['len_feature_chunk']
        self.vnames_cwise = self._get_vnames_cwise()

        self.index_list = list(range(len(self.vnames)))
        if 'train' in self.mode:
            np.random.shuffle(self.index_list)
        self.start_idx = 0

    def _filter_vnames(self, annts):
        string_print = '(vnames) %s: %d -> ' % (self.mode, len(self.vnames))
        vnames_filtered = []
        for v in self.vnames:
            if v not in annts:
                continue
            if not annts[v]['annotations']:
                del self.features[v]
            else:
                vnames_filtered.append(v)

        self.vnames = vnames_filtered
        print (string_print + str(len(self.vnames)))

    def _get_labels(self, annts):
        num_class = len(self.cnames)
        labels = {}
        for v in self.vnames:
            labels[v] = np.zeros((num_class), dtype=np.float32)
            list_l = []
            for a in annts[v]['annotations']:
                list_l.append(a['label'])

            labels[v][[self.cnames.index(l) for l in set(list_l)]] = 1

        return labels

    def _get_annts_cwise(self, annts):
        annts_cwise = {}
        for i, v in enumerate(self.vnames):
            for a in annts[v]['annotations']:
                cn = a['label']
                if cn not in annts_cwise:
                    annts_cwise[cn] = []
                annts_cwise[cn].append([i, a['segment'][0], a['segment'][1]])

        return annts_cwise

    def _get_vnames_cwise(self):
        vnames_cwise = []
        for c in self.cnames:
            vnames_cwise.append([v for v in self.vnames if self.labels[v][self.cnames.index(c)]])

        return vnames_cwise

    def _preprocess(self, features):
        len_features = features.shape[0]
        if len_features >= self.len_snippet:
            start_idx = np.random.randint(len_features-self.len_snippet+1)
            return features[np.arange(start_idx,start_idx+self.len_snippet)], self.len_snippet
        else:
            return np.pad(features[np.arange(len_features)], ((0,self.len_snippet-len_features), (0,0)), mode='constant', constant_values=0), len_features

    def load_data_train(self):
        if self.start_idx+self.batch > len(self.index_list):
            self.start_idx = 0
            np.random.shuffle(self.index_list)

        features = []
        len_features = []
        labels = []
        for i in self.index_list[self.start_idx:self.start_idx+self.batch]:
            v = self.vnames[i]
            f, l = self._preprocess(self.features[v])
            features.append(f)
            len_features.append(l)
            labels.append(self.labels[v])

        self.start_idx += self.batch

        return np.array(features), np.array(labels), len_features

    def reset_start(self):
        self.start_idx = 0

    def load_data_val(self):
        if self.start_idx == self.get_num_videos():
            raise ValueError('check the number of videos')

        v = self.vnames[self.start_idx]
        features = self.features[v]
        labels = self.labels[v]

        self.start_idx += 1

        return np.array([features]), np.array(labels)

    def get_annts_classwise(self):
        return self.annts_cwise

    def get_set_ambiguous(self):
        return self.set_ambiguous

    def get_feature_rate(self):
        return self.fps_extracted / self.len_feature_chunk

    def get_cnames(self):
        return self.cnames

    def get_vnames(self):
        return self.vnames

    def get_num_classes(self):
        return len(self.cnames)

    def get_num_videos(self):
        return len(self.vnames)
