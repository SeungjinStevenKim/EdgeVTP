"""Load EdgeVTP checkpoints and run scene-level trajectory prediction."""

import os

import torch
import yaml
from torch_geometric.data.batch import Batch as tgb

import utils.network as net
import utils.util as ut
from utils.video.traj_clip import clip_scene_predictions


def resolve_device(device_name):
    device_name = str(device_name)
    if device_name.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_name)
    if (
        device_name.startswith("mps")
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def resolve_edgevtp_device(device_name, yolo_device):
    """Avoid MPS+CPU PyG fallback thrash: YOLO on GPU, EdgeVTP on CPU by default on Mac."""
    device_name = str(device_name)
    if device_name == "auto":
        if yolo_device.type == "mps":
            return torch.device("cpu")
        return yolo_device
    return resolve_device(device_name)


class EdgeVTPInference:
    """Wrap model load + single-scene inference for live / video use."""

    def __init__(self, config_path, checkpoint_path=None, dataset_name=None, device="auto"):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.obs_len = self.config["input_data"]["observed_steps"]
        self.pred_len = self.config["input_data"]["prtediction_step"]
        input_data = self.config["input_data"]
        self.relation_limit = input_data.get("relation_neighbor_limit_meter")
        if self.relation_limit is not None:
            self.relation_limit = float(self.relation_limit)
        self.relation_k = input_data.get("relation_neighbor_limit_k")
        if self.relation_k is not None:
            self.relation_k = int(self.relation_k)

        output_size = (
            self.config["input_data"]["points_per_position"]
            * self.config["input_data"]["prtediction_step"]
        )
        num_features = (
            self.config["input_data"]["points_per_position"]
            * self.config["input_data"]["observed_steps"]
        )

        self.device = resolve_device(
            self.config["training"].get("device", device) if device == "auto" else device
        )
        if self.device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.config["training"]["device"] = str(self.device)

        self.model = net.NetGINConv_ve(num_features, output_size, self.config).to(self.device)
        self.output_type = self.config["training"].get("output_type", "mlp")
        self.use_chunked = self.config["training"].get("use_chunked", False)
        self.use_residual = self.config["training"].get("use_residual_separation", False)

        datasets = self.config["input_data"]["dataset"]
        if dataset_name is None:
            dataset_name = datasets[0] if datasets else "ngsim"
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.config["training"]["model_dir"], f"{dataset_name}.pt"
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if (
            "output_layer.weight" not in ckpt
            and "kan_output.layers.0.base_weight" not in ckpt
            and "cp_embed.weight" not in ckpt
        ):
            self.model.legacy_output_passthrough = True
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    def configure_for_coord_mode(self, coord_mode):
        """Carolinas pixel coords use radius limit like main.py (30 ≈ 30 px)."""
        del coord_mode  # scale/homography keep config defaults unchanged

    def _forward_batch(self, obs):
        """Run model on a batch; returns absolute predictions before traj_clip."""
        obs_traj = obs.unsqueeze(1)
        obs_traj_rel = torch.zeros_like(obs_traj)
        obs_traj_rel[:, :, 1:, :] = obs_traj[:, :, 1:, :] - obs_traj[:, :, :-1, :]

        seq_start_end = torch.tensor([[0, obs.shape[0]]], dtype=torch.long)
        data_list = ut.getGraphDataList(
            obs_traj,
            obs_traj_rel,
            seq_start_end,
            relation_neighbor_limit_meter=self.relation_limit,
            relation_neighbor_limit_k=self.relation_k if self.relation_limit is None else None,
        )
        graph_batch = tgb.from_data_list(data_list)
        num_edges = int(graph_batch.edge_index.shape[1]) if graph_batch.edge_index is not None else 0
        if num_edges == 0 and obs.shape[0] > 1:
            k = self.relation_k or 16
            data_list = ut.getGraphDataList(
                obs_traj,
                obs_traj_rel,
                seq_start_end,
                relation_neighbor_limit_k=min(int(k), obs.shape[0] - 1),
            )
            graph_batch = tgb.from_data_list(data_list)
            num_edges = int(graph_batch.edge_index.shape[1]) if graph_batch.edge_index is not None else 0

        edge_weight = getattr(graph_batch, "edge_attr", None)
        if edge_weight is not None and edge_weight.numel() > 0:
            edge_weight = edge_weight.to(self.device)
        else:
            edge_weight = None

        start_pos = obs_traj[:, :, -1, :].squeeze(1)
        if self.use_residual:
            pred_traj = self.model.infer(
                obs_traj_rel,
                graph_batch.x.to(self.device),
                graph_batch.edge_index.to(self.device),
                seq_len=self.pred_len,
                edge_weight=edge_weight,
                start_pos=start_pos,
                x_cx=graph_batch.x_cx.to(self.device),
                x_delta=graph_batch.x_delta.to(self.device),
            )
        else:
            pred_traj = self.model.infer(
                obs_traj_rel,
                graph_batch.x.to(self.device),
                graph_batch.edge_index.to(self.device),
                seq_len=self.pred_len,
                edge_weight=edge_weight,
                start_pos=start_pos,
            )

        if self.use_chunked:
            pred_abs = ut.relative_to_abs(pred_traj, start_pos)
        elif self.output_type in ("bezier", "one_shot_bezier"):
            if self.output_type == "bezier":
                pred_abs = pred_traj
            else:
                pred_abs = ut.relative_to_abs(pred_traj, start_pos)
        else:
            if pred_traj.ndim == 3 and pred_traj.shape[-1] != 2:
                pred_traj = pred_traj.reshape(pred_traj.shape[0], self.pred_len, 2)
            pred_abs = ut.relative_to_abs(pred_traj, start_pos)
        return pred_abs, num_edges

    @torch.no_grad()
    def predict_scene(self, obs_abs, isolate_agents=False):
        if obs_abs.shape[0] == 0:
            return torch.empty((0, self.pred_len, 2))

        obs = torch.as_tensor(obs_abs, dtype=torch.float32, device=self.device)
        if obs.ndim != 3 or obs.shape[1] != self.obs_len or obs.shape[2] != 2:
            raise ValueError(
                f"obs_abs must be (N, {self.obs_len}, 2), got {tuple(obs.shape)}"
            )

        if isolate_agents and obs.shape[0] > 1:
            pred_chunks = []
            for i in range(obs.shape[0]):
                chunk, _ = self._forward_batch(obs[i : i + 1])
                pred_chunks.append(chunk)
            pred_abs = torch.cat(pred_chunks, dim=0)
            num_edges = 0
        else:
            pred_abs, num_edges = self._forward_batch(obs)

        self.last_scene_debug = {
            "num_nodes": int(obs.shape[0]),
            "num_edges": num_edges,
            "isolated": bool(isolate_agents and obs.shape[0] > 1),
        }

        pred_abs = clip_scene_predictions(
            obs.detach().cpu().numpy(),
            pred_abs.detach().cpu().numpy(),
            self.pred_len,
            self.obs_len,
        )
        return torch.as_tensor(pred_abs, dtype=torch.float32)
