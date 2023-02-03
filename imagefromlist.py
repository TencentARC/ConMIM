# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from PIL import Image
import torch.utils.data as data
import sys

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist, root=None):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            if (root is not None):
                impath = os.path.join(root, impath)
            imlist.append((impath, int(imlabel)))

    return imlist



class ImageFromList(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.samples = flist_reader(flist, root=root)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.samples[index]
        try:
            for i in range(5):
                try:
                    img = self.loader(impath)
                    break
                except:
                    continue
        except:
            sys.exit(1)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.samples)
