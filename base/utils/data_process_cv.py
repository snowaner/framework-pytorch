#! /usr/bin/env python
#################################################################################
#     File Name           :     preprocessor.py
#     Created By          :     WanCQ
#     Creation Date       :     [2018-12-02 04:18]
#     Last Modified       :     [2018-12-02 04:25]
#     Description         :     data process under cv
#################################################################################

import cv2 as cv

class Preprocessor(object):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        if len(self.dataset[index]) == 2:
            fname, label = self.dataset[index]
        else:
            fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = cv.imread(fpath) # change PIL Image to Opencv method
        img = img[:,:,(2,0,1)] # change BGR into RGB
        if self.transform is not None:
            img = self.transform(img)
        if len(self.dataset[index]) == 2:
            return img, fname, label
        else:
            return img, fname, pid, camid
