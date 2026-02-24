import torch
import time
import os
import argparse
from statistics import mean
from thop import profile
from torch_geometric.data.batch import Batch as tgb
import utils.loader as dl
import utils.network as net
import utils.util as ut
import yaml
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Training and validation parameters.')
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Optional profile switch so a single pedestrian YAML can drive multiple models.
selected_model = config.get('training', {}).get('selected_model')
model_profiles = config.get('training', {}).get('model_profiles')
if selected_model and isinstance(model_profiles, dict):
    if selected_model not in model_profiles:
        raise ValueError(f"Unknown selected_model '{selected_model}'. Available: {list(model_profiles.keys())}")
    for key, value in model_profiles[selected_model].items():
        config['training'][key] = value

relation_neighbor_limit_meter = config['input_data'].get('relation_neighbor_limit_meter')
if relation_neighbor_limit_meter is not None:
    relation_neighbor_limit_meter = float(relation_neighbor_limit_meter)
    
output_size = config['input_data']['points_per_position']*config['input_data']['prtediction_step']
num_features = config['input_data']['points_per_position']*config['input_data']['observed_steps']

# make sure to change name of the datasets models and this line
if config['input_data']['dataset'] == ["VIRAT_ActEV"]:
    delim = 'space'
else:
    delim = 'tab'


# if config['training']['save_model']:
#     if not os.path.exists(f"models/{config['training']['save_folder']}/"):
#             os.makedirs(f"models/{config['training']['save_folder']}/")

current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
experiment_root = config['training'].get('experiment_root', 'experiments/pedestrian')
run_name = config['training'].get('run_name', current_time)
mode_dir = "train" if config['training']['train'] else "inference"
dir_name = os.path.join(experiment_root, mode_dir, run_name)
os.makedirs(dir_name, exist_ok=True)

with open(f"{dir_name}/config.yaml", 'w') as config_file:
    yaml.dump(config, config_file)
    
if config['training']['train']:
    lr = config['training']['learning_rate']
    for test_file in config['input_data']['dataset']:
        writer = SummaryWriter(f"{dir_name}/{test_file}")
        print("Test file: " + test_file)

        device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv_ve(num_features, output_size, config).to(device)
        
        # Print total parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params}")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['training']['milestones'], gamma=0.1)

        data_dir = 'datasets/'+test_file+'/'
        augment = config['input_data'].get('augment', False)
        augment_mode = config['input_data'].get('augment_mode', 'full')
        augment_rotation = config['input_data'].get('augment_rotation', None)
        _, train_loader = dl.data_loader(data_dir+config['input_data']['train_folder'],
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim,
                                        augment=augment,
                                        augment_mode=augment_mode,
                                        augment_rotation=augment_rotation)

        _, val_loader = dl.data_loader(data_dir+config['input_data']['val_folder'],
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim,
                                        augment=False)

        
        best_info = [1000.0, 1000.0, 0]
        for epoch in range(0, config['training']['epoches']):
            print(f"----------Epoch #{epoch}----------")
            print("-Training-")
            losses = ut.train(
                model,
                train_loader,
                optimizer,
                device,
                obs_step=config['input_data']['observed_steps'],
                relation_neighbor_limit_meter=relation_neighbor_limit_meter,
                output_type=config['training'].get('output_type', 'mlp'),
                use_chunked=config['training'].get('use_chunked', False),
            )
            writer.add_scalar('Train_Loss', np.mean(losses), global_step=epoch)
            if(epoch%config['training']['validation_interval']==0):
                ade, fde, test_losses = ut.test(
                    model,
                    val_loader,
                    device,
                    obs_step=config['input_data']['observed_steps'],
                    relation_neighbor_limit_meter=relation_neighbor_limit_meter,
                    output_type=config['training'].get('output_type', 'mlp'),
                    use_chunked=config['training'].get('use_chunked', False),
                )
                writer.add_scalar('Test_Loss', np.mean(test_losses), global_step=epoch)
                writer.add_scalar('ADE', ade, global_step=epoch)
                writer.add_scalar('FDE', fde, global_step=epoch)
                print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
                if ade < best_info[0]:
                    best_info[0] = ade
                    best_info[1] = fde
                    best_info[2] = epoch
                    if (config['training']['save_model']):
                        model_path = f"{dir_name}/{test_file}.pt"
                        torch.save(model.state_dict(), model_path)
            scheduler.step()  # LR decay every epoch (milestones)
        print(test_file + "   Best ADE: " + str(best_info[0]) + "   FDE: " + str(best_info[1]) + "   Epoch: " + str(best_info[2]) + "   lr: " + str(lr) + "\n\n")
        writer.close()

else:
    with torch.no_grad():
        print("=" * 58)
        print(f"Experiment: {run_name}")
        print(f"Results saved to: {dir_name}")
        print("=" * 58)
        print(f"Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        model_type = config['training'].get('selected_model', 'original')
        output_type = config['training'].get('output_type', 'mlp')
        print(f"Model: VT-Former ({model_type}), output_type: {output_type}")
        print("")

        all_ops = []
        times = []
        res = []
        frame_list = []
        total_params_k = None

        
        for test_file in config['input_data']["dataset"]:
            ls = []
            total_traj = 0
            all_normal_RMSE = []
            device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
            model = net.NetGINConv_ve(num_features, output_size, config).to(device)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if total_params_k is None:
                total_params_k = total_params / 1000.0

            model_folder = 'models/TRAINED/'
            ckpt_path = os.path.join(config['training']['model_dir'], test_file) + ".pt"
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # Older checkpoints may not have output_norm/output layers.
            if (
                'output_layer.weight' not in ckpt
                and 'kan_output.layers.0.base_weight' not in ckpt
                and 'cp_embed.weight' not in ckpt
            ):
                model.legacy_output_passthrough = True
            model.load_state_dict(ckpt, strict=False)
            data_dir = 'datasets/'+test_file+'/'



            if config['input_data']['test_folder'] != 'None':
                _, test_loader = dl.data_loader(data_dir+config['input_data']['test_folder'], 
                                                batch_size=1,
                                                obs_len=config['input_data']['observed_steps'],
                                                pred_len=config['input_data']['prtediction_step'],
                                                delim=delim)
            else:
                _, test_loader = dl.data_loader(data_dir+config['input_data']['val_folder'], 
                                                batch_size=1,
                                                obs_len=config['input_data']['observed_steps'],
                                                pred_len=config['input_data']['prtediction_step'],
                                                delim=delim)

            ade_batches, fde_batches = [], []
            rmse_batch = torch.zeros(config['input_data']['prtediction_step']).to(device)
            number_of_traj = 0

            model.eval()
            for batch in tqdm(test_loader):
                batch = [tensor.to(device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                    non_linear_ped, loss_mask, seq_start_end, frame_id) = batch
                
                total_traj += pred_traj_gt.size(0)

                data_list = ut.getGraphDataList(
                    obs_traj,
                    obs_traj_rel,
                    seq_start_end,
                    relation_neighbor_limit_meter=relation_neighbor_limit_meter,
                )
                graph_batch = tgb.from_data_list(data_list)
                edge_weight = getattr(graph_batch, 'edge_attr', None)
                edge_weight = edge_weight.to(device) if edge_weight is not None and edge_weight.numel() > 0 else None

                output_type = config['training'].get('output_type', 'mlp')
                use_chunked = config['training'].get('use_chunked', False)
                start_pos = obs_traj[:, :, -1, :].squeeze(1) if (output_type == 'bezier' or output_type == 'one_shot_bezier' or use_chunked) else None
                start = time.time()
                if getattr(model, 'use_residual_separation', False):
                    pred_traj = model.infer(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device), seq_len=test_loader.dataset.pred_len, edge_weight=edge_weight, start_pos=start_pos, x_cx=graph_batch.x_cx.to(device), x_delta=graph_batch.x_delta.to(device))
                else:
                    pred_traj = model.infer(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device), seq_len=test_loader.dataset.pred_len, edge_weight=edge_weight, start_pos=start_pos)
                end = time.time()
                times.append(end-start)

                if use_chunked:
                    pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:, :, -1, :].squeeze(1))
                elif output_type == 'bezier':
                    pred_traj_real = pred_traj
                else:
                    pred_traj = pred_traj[:, :, :]
                    pred_traj = pred_traj.reshape(pred_traj.shape[0], test_loader.dataset.pred_len, 2)
                    pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:, :, -1, :].squeeze(1))

                ade_batches.append(torch.sum(ut.displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
                fde_batches.append(torch.sum(ut.final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())
        
                rmse_batch += (ut.rmse(pred_traj_real, pred_traj_gt, mode='raw')).detach()
                number_of_traj+=1
                all_normal_RMSE.append((ut.rmse(pred_traj_real.to('cpu'), pred_traj_gt.to('cpu'), mode='raw')))
                
                

                # ops, params = profile(model, inputs=(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device)))
                # all_ops.append(ops)
            # all_normal_RMSE = all_normal_RMSE.to('cpu')
            array_list = np.array([tensor.numpy() for tensor in all_normal_RMSE])
            # Save the list of NumPy arrays to a text file
            file_path = '/home/adaneshp/PishguVe_Tran/anomaly_res/All_normal_ADE_FDE_Left_Right_list_data.txt' 
            # with open(file_path, 'w') as file:
            #     # for array in array_list:
            #     np.savetxt(file, array_list, delimiter=',', fmt='%0.3f')
            ade = sum(ade_batches) / (total_traj * config['input_data']['prtediction_step'])
            fde = sum(fde_batches) / (total_traj)

            rmse = rmse_batch/number_of_traj
            avg_rmse = torch.mean(rmse).item()

            # Pedestrian snapshots: 1.2s(F3), 2.4s(F6), 3.6s(F9), 4.8s(F12)
            # Vehicle snapshots: 1s(F5), 2s(F10), 3s(F15), 4s(F20), 5s(F25)
            if config['input_data']['subject'] == 'pedestrian':
                f3, f6, f9, f12 = rmse[2].item(), rmse[5].item(), rmse[8].item(), rmse[11].item()
                snap_arr = np.array([f3, f6, f9, f12])
                ls.append([f3, f6, f9, f12])
                print(f"{test_file}  ADE: {ade}  FDE: {fde}")
                print("RMSE: 1.2s, 2.4s, 3.6s, 4.8s (approx.)")
                print(snap_arr)
            else:
                f1 = rmse[4].item()
                f2 = rmse[9].item()
                f3 = rmse[14].item()
                f4 = rmse[19].item()
                f5 = rmse[24].item()
                snap_arr = np.array([f1, f2, f3, f4, f5])
                ls.append([f1, f2, f3, f4, f5])
                print(f"{test_file}  ADE: {ade}  FDE: {fde}")
                print("RMSE: 1s, 2s, 3s, 4s, 5s (approx.)")
                print(snap_arr)

            ls.append(test_file)
            ls.append(ade)
            ls.append(fde)
            ls.append(avg_rmse)
            res.append(ls)
        avg_fde = 0
        avg_ade = 0
        avg_rmse_total = 0
        avg_snapshots = None

        for data_st in res:
            snapshots = data_st[0]
            if avg_snapshots is None:
                avg_snapshots = [0.0] * len(snapshots)
            for i in range(len(snapshots)):
                avg_snapshots[i] += snapshots[i]
            
            avg_ade = avg_ade + data_st[2]
            avg_fde = avg_fde + data_st[3]
            avg_rmse_total = avg_rmse_total + data_st[4]
            
        avg_ade = avg_ade/len(res)
        avg_fde = avg_fde/len(res)
        avg_rmse_total = avg_rmse_total/len(res)
        for i in range(len(avg_snapshots)):
            avg_snapshots[i] /= len(res)

        print("Average Execution Time: " + str(mean(times)) + " sec")

        method_name = config['training'].get('run_name', 'VT-Former')
        if config['input_data']['subject'] == 'pedestrian':
            s1, s2, s3, s4 = avg_snapshots[0], avg_snapshots[1], avg_snapshots[2], avg_snapshots[3]
            params_str = f"{total_params_k:.1f}" if total_params_k else "-"
            print("")
            print("==============================")
            print("FINAL SUMMARY ROW (Average across all datasets)")
            print("Method | 1.2s | 2.4s | 3.6s | 4.8s | ADE | FDE | Params(K)")
            print(f"{method_name} | {s1:.2f} | {s2:.2f} | {s3:.2f} | {s4:.2f} | {avg_ade:.2f} | {avg_fde:.2f} | {params_str}")
            print("==============================")
        else:
            s1, s2, s3, s4, s5 = avg_snapshots[0], avg_snapshots[1], avg_snapshots[2], avg_snapshots[3], avg_snapshots[4]
            params_str = f"{total_params_k:.1f}" if total_params_k else "-"
            print("")
            print("==============================")
            print("FINAL SUMMARY ROW (Average across all datasets)")
            print("Method | 1s | 2s | 3s | 4s | 5s | ADE | FDE | Params(K)")
            print(f"{method_name} | {s1:.2f} | {s2:.2f} | {s3:.2f} | {s4:.2f} | {s5:.2f} | {avg_ade:.2f} | {avg_fde:.2f} | {params_str}")
            print("==============================")

        print([[d[1], d[2], d[3]] for d in res])
        print("Average ADE: ", str(avg_ade))
        print("Average FDE: ", str(avg_fde))

        # Save results to file
        results_path = os.path.join(dir_name, 'inference_results.txt')
        with open(results_path, 'w') as f:
            f.write(f"Experiment: {method_name}\n")
            f.write(f"Model dir: {config['training']['model_dir']}\n")
            f.write(f"Relation limit: {relation_neighbor_limit_meter}m\n\n")
            for d in res:
                f.write(f"{d[1]}  ADE: {d[2]:.4f}  FDE: {d[3]:.4f}\n")
            f.write(f"\nAverage ADE: {avg_ade:.4f}\n")
            f.write(f"Average FDE: {avg_fde:.4f}\n")
            if config['input_data']['subject'] == 'pedestrian':
                f.write(f"\nSummary | 1.2s | 2.4s | 3.6s | 4.8s | ADE | FDE\n")
                f.write(f"{method_name} | {s1:.2f} | {s2:.2f} | {s3:.2f} | {s4:.2f} | {avg_ade:.2f} | {avg_fde:.2f}\n")
            else:
                f.write(f"\nSummary | 1s | 2s | 3s | 4s | 5s | ADE | FDE\n")
                f.write(f"{method_name} | {s1:.2f} | {s2:.2f} | {s3:.2f} | {s4:.2f} | {s5:.2f} | {avg_ade:.2f} | {avg_fde:.2f}\n")
        print(f"\nResults saved to: {results_path}")
