import numpy as np
import random
from os.path import join
from torch.utils.data import Dataset
from PIL import Image
import ldm.data.vfitransforms as vt
from functools import partial


class HOI_triplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True):
        seq_dir = join(db_dir, 'images')
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t

        # TODO : Generate these text files
        if train:
            seq_list_txt = join(db_dir, 'sep_trainlist.txt')
        else:
            seq_list_txt = join(db_dir, 'sep_testlist.txt')

        with open(seq_list_txt) as f:
            contents = f.readlines()
            seq_path = [line.strip() for line in contents if line != '\n']

        self.seq_path_list = [join(seq_dir, *line.split('/')) for line in seq_path]

    def __getitem__(self, index):
        rawFrame1 = Image.open(join(self.seq_path_list[index],  "img1.png"))
        rawFrame2 = Image.open(join(self.seq_path_list[index],  "img2.png"))
        rawFrame3 = Image.open(join(self.seq_path_list[index],  "img3.png"))

        if self.crop_sz is not None:
            rawFrame1, rawFrame2, rawFrame3 = vt.rand_crop(rawFrame1, rawFrame2, rawFrame3, sz=self.crop_sz)

        if self.augment_s:
            rawFrame1, rawFrame2, rawFrame3 = vt.rand_flip(rawFrame1, rawFrame2, rawFrame3, p=0.5)
        
        if self.augment_t:
            rawFrame1, rawFrame2, rawFrame3 = vt.rand_reverse(rawFrame1, rawFrame2, rawFrame3, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame1, frame2, frame3 = map(to_array, (rawFrame1, rawFrame2, rawFrame3)) #(256,256,3), 0-255

        frame1 = frame1/127.5 - 1.0
        frame2 = frame2/127.5 - 1.0
        frame3 = frame3/127.5 - 1.0

        return {'image': frame2, 'prev_frame': frame1, 'next_frame': frame3}

    def __len__(self):
        return len(self.seq_path_list)

class HOI_Train_triplet(Dataset):
    def __init__(self, db_dir, crop_sz=[256,256], p_datasets=None, iter=False, samples_per_epoch=1000):
        hoi_train = HOI_triplet(join(db_dir, 'hoi-triplets'), train=True,  crop_sz=crop_sz)

        self.datasets = [hoi_train]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch