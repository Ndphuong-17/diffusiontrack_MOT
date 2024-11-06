from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
from torch import utils
import torch


class DiffMOTDataset(Dataset):
    def __init__(self, path, config=None):
        self.config = config
        try:
            self.interval = self.config.interval + 1
        except:
            self.interval = 4 + 1

        self.trackers = {}
        self.images = {}
        self.nframes = {}
        self.ntrackers = {}

        self.nsamples = {}
        self.nS = 0

        self.nds = {}
        self.cds = {}

        self.image_dims = {}

        if os.path.isdir(path):
            if 'MOT' in path:
                self.seqs = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13", "MOT20-01", "MOT20-02", "MOT20-03", "MOT20-05"]
            else:
                self.seqs = [s for s in os.listdir(path) if not s.startswith('.') and "gt_t" not in s]
            self.seqs.sort()
            lastindex = 0
            for seq in self.seqs:
                trackerPath = os.path.join(path, seq, "img1/*.txt")
                self.trackers[seq] = sorted(glob.glob(trackerPath))
                self.ntrackers[seq] = len(self.trackers[seq])
                if 'MOT' in seq:
                    imagePath = os.path.join(path, '../../images/train', seq, "img1/*.*")
                else:
                    imagePath = os.path.join('/'.join(sum([d.split('/') for d in path.split('\\')], [])[:-1]) + "/" + seq, "img1/*.*")

                    
                self.images[seq] = sorted(glob.glob(imagePath))
                self.nframes[seq] = len(self.images[seq])

                if self.images[seq]:  # Ensure there is at least one image
                    img = Image.open(self.images[seq][0])
                    width, height = img.size
                    self.image_dims[seq] = (width, height)

                self.nsamples[seq] = {}
                for i, pa in enumerate(self.trackers[seq]):
                    self.nsamples[seq][i] = np.loadtxt(pa, dtype=np.float32).shape[0] - self.interval
                    # self.nsamples[seq][i] = len(np.loadtxt(pa, dtype=np.float32).reshape(-1,7)) - self.interval
                    self.nS += self.nsamples[seq][i]


                self.nds[seq] = [x for x in self.nsamples[seq].values()]
                self.cds[seq] = [sum(self.nds[seq][:i]) + lastindex for i in range(len(self.nds[seq]))]
                lastindex = self.cds[seq][-1] + self.nds[seq][-1]

        # print('=' * 80)
        # print('dataset summary')
        print(self.nS)
        print('=' * 80)



    def __getitem__(self, files_index):

        for i, seq in enumerate(self.cds):
            if files_index >= self.cds[seq][0]:
                ds = seq
                for j, c in enumerate(self.cds[seq]):
                    if files_index >= c:
                        trk = j
                        start_index = c
                    else:
                        break
            else:
                break

        track_path = self.trackers[ds][trk]
        track_gt = np.loadtxt(track_path, dtype=np.float32)

        init_index = files_index - start_index

        cur_index = init_index + self.interval
        cur_gt = track_gt[cur_index]
        cur_bbox = cur_gt[2:6]

        boxes = [track_gt[init_index + tmp_ind][2:6] for tmp_ind in range(self.interval)]
        delt_boxes = [boxes[i+1] - boxes[i] for i in range(self.interval - 1)]
        conds = np.concatenate((np.array(boxes)[1:], np.array(delt_boxes)), axis=1)

        delt = cur_bbox - boxes[-1]

        width, height = self.image_dims[ds]
        image_path = self.images[ds][cur_index]

        if len(cur_gt) >= 73:
            feat = torch.Tensor(cur_gt[-64:])
        else:
            feat = []
            



        ret = {
            "cur_gt": cur_gt, 
            "cur_bbox": cur_bbox, 
            "condition": conds, 
            "delta_bbox": delt,
            "width": width,
            "height": height,
            "image_path": image_path,
            "feat": feat
        }

        return ret

    def __len__(self):
        return self.nS
    
    

class DiffMOTDataLoader(utils.data.DataLoader):
    def __init__(self, dataset, config):
        super().__init__(
            dataset=dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.preprocess_workers,
            pin_memory=True
        )

if __name__ == "__main__":
    data_path = r"Data\DanceTrack\train"
    a = DiffMOTDataset(path = data_path)
    print(a[700])
    pass
