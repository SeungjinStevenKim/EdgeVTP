#util.py
import torch
import os
import numpy as np
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch as tgb
from tqdm import tqdm

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (batch, seq_len, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (batch, seq_len, 2)
    """
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj

def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    # pred_traj: (B, L, 2), pred_traj_gt: (B, 1, L, 2)
    gt = pred_traj_gt.squeeze(1)
    loss = (loss_mask.unsqueeze(dim=2) * (gt - pred_traj)**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def rmse(pred_traj, pred_traj_gt, mode='raw'):
    # pred_traj: (B, L, 2), pred_traj_gt: (B, 1, L, 2)
    gt = pred_traj_gt.squeeze(1)
    loss = (gt - pred_traj)**2
    # Returns (L,) averaged over batch
    return torch.sqrt(loss.sum(dim=2).mean(dim=0))

def displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    gt = pred_traj_gt.squeeze(1)
    loss = (gt - pred_traj)**2
    # Sum of distances over time for each agent
    dist = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(dist)
    return dist

def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    # pred_pos: (B, 2), pred_pos_gt: (B, 2)
    dist = torch.sqrt(torch.sum((pred_pos_gt - pred_pos)**2, dim=1))
    if mode == 'sum':
        return torch.sum(dist)
    return dist

def getGraphDataList(obs_traj, obs_traj_rel, seq_start_end, relation_neighbor_limit_meter=None):
    data_list = []
    for (start, end) in seq_start_end:
        x1=obs_traj[start:end,:,:,:].reshape(end-start, int(obs_traj.shape[2]*obs_traj.shape[3]))
        x2=obs_traj_rel[start:end,:,:,:].reshape(end-start, int(obs_traj_rel.shape[2]*obs_traj_rel.shape[3]))
        x = torch.cat((x1,x2),dim=1)
        NUM_NODES = x.shape[0]
        if relation_neighbor_limit_meter is None:
            edge_list1, edge_list2 = [], []
            for n in range(NUM_NODES):
                for k in range(NUM_NODES):
                    if k != n:
                        edge_list1.append(n); edge_list2.append(k)
            edge_index = torch.tensor([edge_list1, edge_list2], dtype=torch.long)
        else:
            last_pos = obs_traj[start:end, 0, -1, :]
            pos_np = last_pos.detach().cpu().numpy()
            tree = cKDTree(pos_np)
            src_list, dst_list = [], []
            for i in range(NUM_NODES):
                neighbors = tree.query_ball_point(pos_np[i], r=relation_neighbor_limit_meter)
                for j in neighbors:
                    if j != i:
                        src_list.append(i); dst_list.append(j)
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.empty((2, 0), dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index.to(obs_traj.device), num_nodes=NUM_NODES))
    return data_list

def train(model, train_loader, optimizer, device, obs_step, relation_neighbor_limit_meter=None, output_type='mlp', use_chunked=False):
    losses = []
    model.train()
    chunk_size = getattr(model, 'chunk_size', 5)
    for batch in tqdm(train_loader):
        batch = [tensor.to(device) for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, loss_mask, seq_start_end, _ ) = batch
        optimizer.zero_grad()
        
        start_pos = obs_traj[:, :, -1, :].squeeze(1)
        if use_chunked:
            pred_flat = pred_traj_gt.squeeze(1); pred_rel_flat = pred_traj_gt_rel.squeeze(1)
            num_chunks = pred_flat.size(1) // chunk_size
            tgt = torch.cat([pred_flat, pred_rel_flat], dim=-1).view(pred_flat.size(0), num_chunks, chunk_size, 4)
        else:
            tgt = torch.cat((pred_traj_gt, pred_traj_gt_rel), dim=1).view(pred_traj_gt.size(0), pred_traj_gt.size(2), -1)
        
        graph_batch = tgb.from_data_list(getGraphDataList(obs_traj, obs_traj_rel, seq_start_end, relation_neighbor_limit_meter))
        pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device), tgt.to(device), start_pos=start_pos)
        
        # All models now output relative dx, dy (B, L, 2)
        pred_traj_real = relative_to_abs(pred_traj, start_pos)
        
        loss = l2_loss(pred_traj_real, pred_traj_gt, loss_mask[:, obs_step:], mode='average')
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return losses

def test(model, test_loader, device, obs_step, relation_neighbor_limit_meter=None, output_type='mlp', use_chunked=False):
    total_traj = 0
    ade_batches, fde_batches = [], []
    model.eval()
    for batch in tqdm(test_loader):
        batch = [tensor.to(device) for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, _, _, loss_mask, seq_start_end, _) = batch
        total_traj += pred_traj_gt.size(0)
        start_pos = obs_traj[:, :, -1, :].squeeze(1)
        
        graph_batch = tgb.from_data_list(getGraphDataList(obs_traj, obs_traj_rel, seq_start_end, relation_neighbor_limit_meter))
        pred_traj = model.infer(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device), seq_len=test_loader.dataset.pred_len, start_pos=start_pos)
        
        pred_traj_real = relative_to_abs(pred_traj, start_pos)
        
        ade_batches.append(displacement_error(pred_traj_real, pred_traj_gt, mode='sum').item())
        fde_batches.append(final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='sum').item())
        
    ade = sum(ade_batches) / (total_traj * test_loader.dataset.pred_len)
    fde = sum(fde_batches) / total_traj
    return ade, fde, []
