# This file is based on the following git repository: https://github.com/agrimgupta92/sgan

# It loads the dataset presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from tqdm import tqdm

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, frame_id_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    pred_traj = torch.cat(pred_seq_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    frame_id = torch.cat(frame_id_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, seq_start_end, frame_id
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def _augment_scene(obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, prob_rotate=0.5, prob_flip=0.5, scale_range=(0.95, 1.05), mode='full', augment_rotation=None):
    """
    Apply random augmentation to a scene. Same transform for all agents (shared coordinate frame).
    obs_traj: [N, 2, obs_len], pred_traj: [N, 2, pred_len]

    Args:
        mode: 'full' = rotation + flip + scale (pedestrian). 'scale_only' = scale only (vehicle-safe).
        augment_rotation: If set (float, degrees), allows limited rotation even in 'scale_only' mode.
                          Range will be [-augment_rotation, +augment_rotation].
    """
    obs_traj = obs_traj.clone()
    pred_traj = pred_traj.clone()

    all_pos = torch.cat([obs_traj, pred_traj], dim=2)  # [N, 2, obs_len+pred_len]
    center = all_pos.mean(dim=(0, 2), keepdim=True)  # [1, 2, 1]

    # Random rotation
    # - In 'full' mode: full 0-2pi rotation
    # - In 'scale_only' mode: only if augment_rotation is specified (limited range)
    should_rotate = False
    rot_min, rot_max = 0, 2 * np.pi
    
    if mode == 'full':
        should_rotate = True
    elif mode == 'scale_only' and augment_rotation is not None and augment_rotation > 0:
        should_rotate = True
        rad = np.radians(augment_rotation)
        rot_min, rot_max = -rad, rad

    if should_rotate and np.random.rand() < prob_rotate:
        angle = np.random.uniform(rot_min, rot_max)
        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([[c, -s], [s, c]], dtype=obs_traj.dtype, device=obs_traj.device)
        obs_traj = (obs_traj - center).permute(0, 2, 1) @ R.T + center.permute(0, 2, 1)
        obs_traj = obs_traj.permute(0, 2, 1)
        pred_traj = (pred_traj - center).permute(0, 2, 1) @ R.T + center.permute(0, 2, 1)
        pred_traj = pred_traj.permute(0, 2, 1)

    # Random flip (x-axis)
    if mode == 'full' and np.random.rand() < prob_flip:
        obs_traj[:, 0, :] = -obs_traj[:, 0, :] + 2 * center[0, 0, 0]
        pred_traj[:, 0, :] = -pred_traj[:, 0, :] + 2 * center[0, 0, 0]

    # Random scale (both full and scale_only)
    if np.random.rand() < 0.5:
        scale = np.random.uniform(*scale_range)
        obs_traj = (obs_traj - center) * scale + center
        pred_traj = (pred_traj - center) * scale + center

    # Recompute relative trajectories
    obs_traj_rel = torch.zeros_like(obs_traj_rel)
    obs_traj_rel[:, :, 1:] = obs_traj[:, :, 1:] - obs_traj[:, :, :-1]
    pred_traj_rel = torch.zeros_like(pred_traj_rel)
    pred_traj_rel[:, :, 0] = pred_traj[:, :, 0] - obs_traj[:, :, -1]
    pred_traj_rel[:, :, 1:] = pred_traj[:, :, 1:] - pred_traj[:, :, :-1]

    return obs_traj, pred_traj, obs_traj_rel, pred_traj_rel


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t', augment=False, augment_mode='full', augment_rotation=None
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - augment: If True, apply augmentation during __getitem__
        - augment_mode: 'full' (rotation+flip+scale, pedestrian) | 'scale_only' (vehicle-safe)
        - augment_rotation: (float) limited rotation angle in degrees for vehicle/scale_only mode
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.augment = augment
        self.augment_mode = augment_mode
        self.augment_rotation = augment_rotation
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        frame_id_list = []
        loss_mask_list = []
        non_linear_ped = []
        for path in tqdm(all_files):
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_frame_id = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_frame_id = np.transpose(curr_ped_seq[:, :2])
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    if curr_ped_seq.shape[1] != pad_end-pad_front: # to check if a sequence has a lost frame
                        continue
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_frame_id[_idx, :, pad_front:pad_end] = curr_ped_frame_id
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    frame_id_list.append(curr_frame_id[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        seq_frame = np.concatenate(frame_id_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.seq_frame_id = torch.from_numpy(seq_frame)

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        obs_traj = self.obs_traj[start:end, :].clone()
        pred_traj = self.pred_traj[start:end, :].clone()
        obs_traj_rel = self.obs_traj_rel[start:end, :].clone()
        pred_traj_rel = self.pred_traj_rel[start:end, :].clone()

        if self.augment:
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = _augment_scene(
                obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
                mode=self.augment_mode, augment_rotation=self.augment_rotation
            )

        out = [
            obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.seq_frame_id[start:end, :]
        ]
        return out
