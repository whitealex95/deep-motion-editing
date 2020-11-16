from torch.utils.data import Dataset
import os
import sys
sys.path.append("../utils")
import numpy as np
import torch
from Quaternions import Quaternions
from option_parser import get_std_bvh


class MotionData(Dataset):
    """
    Clip long dataset into fixed length window for batched training
    each data is a 2d tensor with shape (Joint_num*3) * Time
    """
    def __init__(self, args):
        super(MotionData, self).__init__()
        name = args.character
        file_path = './datasets/{}/{}.npy'.format(args.dataset,name)

        if args.debug:
            file_path = file_path[:-4] + '_debug' + file_path[-4:]

        print('load from file {}'.format(file_path))
        self.total_frame = 0
        self.std_bvh = get_std_bvh(args, args.character)  # get standard bvh for offsets(same for all motions).
        self.args = args
        self.data = []
        self.motion_length = []
        motions = np.load(file_path, allow_pickle=True)  # load motion
        # motions.shape: 7 ( 7 motions per character)
        # motions[0].shape: 18, 75 <- 18: num of frames(Back Squat), 75 = 3 * 25 (euler angle)
        # motions[1].shape: 13, 75 <- 13: num of frames(Drunk Walk Backwards), 75 = 3 * 25 (euler angle)
        # motions[2].shape: 274, 75 <- 274: num of frames(Samba Dancing), 75 = 3 * 25 (euler angle)
        
        motions = list(motions)
        if len(motions) == 1 and motions[0].shape[0] <= 1:  # single motion just used for preprocessing
            # __import__('pdb').set_trace()
            motions[0] = np.concatenate([motions[0]] * args.window_size)
        new_windows = self.get_windows(motions)  # default window size = 64 # split motion into same window_size
        # our motion data
        self.data.append(new_windows)
        self.data = torch.cat(self.data)
        self.data = self.data.permute(0, 2, 1)

        # The data(not only offset, but also the quaternions) is normalized(along time axis) by default!
        if args.normalization == 1:  
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.var = torch.var(self.data, (0, 2), keepdim=True)
            self.var = self.var ** (1/2)
            idx = self.var < 1e-5
            self.var[idx] = 1
            self.data = (self.data - self.mean) / self.var
        else:
            # IMPORTANT  
            # we should not train with normalization because unknown skeleton object does not have mean or variance
            self.mean = torch.mean(self.data, (0, 2), keepdim=True)
            self.mean.zero_()
            self.var = torch.ones_like(self.mean)

        # Single windowed motion will not work due to this! so i erased it
        if False:
            train_len = self.data.shape[0] * 95 // 100
            self.test_set = self.data[train_len:, ...]
            self.data = self.data[:train_len, ...]
        self.data_reverse = torch.tensor(self.data.numpy()[..., ::-1].copy())

        self.reset_length_flag = 0
        self.virtual_length = 0
        print('Window count: {}, total frame (without downsampling): {}'.format(len(self), self.total_frame))

    def reset_length(self, length):
        self.reset_length_flag = 1
        self.virtual_length = length

    def __len__(self):
        if self.reset_length_flag:
            return self.virtual_length
        else:
            return self.data.shape[0]

    def __getitem__(self, item):
        if isinstance(item, int): item %= self.data.shape[0]
        if self.args.data_augment == 0 or np.random.randint(0, 2) == 0:
            return self.data[item]
        else:
            return self.data_reverse[item]

    def get_windows(self, motions):
        new_windows = []
        # __import__('pdb').set_trace()

        for motion in motions:
            self.total_frame += motion.shape[0]  # used only for logging
            # In original paper, 60fps -> 30fps
            # Since our data is based on 30fps, we should NOT subsample
            if self.args.use_original:
                motion = self.subsample(motion)
            else:
                pass
            self.motion_length.append(motion.shape[0])
            step_size = self.args.window_size // 2
            window_size = step_size * 2
            # Problem occurs when window size is bigger than the bvh animation length!
            n_window = motion.shape[0] // step_size - 1
            for i in range(n_window):
                begin = i * step_size
                end = begin + window_size

                new = motion[begin:end, :]  # [64, 75] <- 64: window_size, 75=25(J) x 3(euler)
                if self.args.rotation == 'quaternion':
                    new = new.reshape(new.shape[0], -1, 3)  # [64, 25, 3] <- W, J, 3
                    rotations = new[:, :-1, :]  # [64, 24, 3] <- W, J-1, 3  Why remove?
                    rotations = Quaternions.from_euler(np.radians(rotations)).qs  # [64, 24, 4]
                    rotations = rotations.reshape(rotations.shape[0], -1)  # [64, 96]
                    # positions = new[:, -1, :]  # [64, 3]
                    # positions = np.concatenate((new, np.zeros((new.shape[0], new.shape[1], 1))), axis=2)  # [ 64, 25, 4]
                    new = np.concatenate((rotations, new[:, -1, :].reshape(new.shape[0], -1)), axis=1)  # [64, 99]: rot + position?

                new = new[np.newaxis, ...]

                new_window = torch.tensor(new, dtype=torch.float32)
                new_windows.append(new_window)

        return torch.cat(new_windows)

    def subsample(self, motion):  # motions are subsampled!!
        # In original paper, 60fps -> 30fps
        # Since our data is based on 30fps, we should NOT subsample
        return motion[::2, :]

    def denormalize(self, motion):
        if self.args.normalization:
            if self.var.device != motion.device:
                self.var = self.var.to(motion.device)
                self.mean = self.mean.to(motion.device)
            ans = motion * self.var + self.mean
        else: ans = motion
        return ans
