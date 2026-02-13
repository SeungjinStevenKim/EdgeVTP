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
dir_name = f"models/{config['training']['save_folder']}/{current_time}"
os.makedirs(dir_name)

with open(f"{dir_name}/config.yaml", 'w') as config_file:
    yaml.dump(config, config_file)
    
if config['training']['train']:
    saveFolder = config['training']['save_folder'] 
    lr = config['training']['learning_rate']
    for test_file in config['input_data']['dataset']:
        writer = SummaryWriter(f"{dir_name}/{test_file}")
        print("Test file: " + test_file)

        device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv_ve(num_features, output_size, config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['training']['milestones'], gamma=0.1)

        data_dir = 'datasets/'+test_file+'/'
        _, train_loader = dl.data_loader(data_dir+config['input_data']['train_folder'], 
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim)

        _, val_loader = dl.data_loader(data_dir+config['input_data']['val_folder'], 
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim)

        
        best_info = [1000.0, 1000.0, 0]
        for epoch in range(0, config['training']['epoches']):
            print(f"----------Epoch #{epoch}----------")
            print("-Training-")
            losses = ut.train(model, train_loader, optimizer, device, obs_step=config['input_data']['observed_steps'])
            writer.add_scalar('Train_Loss', np.mean(losses), global_step=epoch)
            if(epoch%config['training']['validation_interval']==0):
                ade, fde, test_losses = ut.test(model, val_loader, device, obs_step=config['input_data']['observed_steps'])
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
        print(test_file + "   Best ADE: " + str(best_info[0]) + "   FDE: " + str(best_info[1]) + "   Epoch: " + str(best_info[2]) + "   lr: " + str(lr) + "\n\n")
        writer.close()

else:
    with torch.no_grad():
        all_ops = []
        times = []
        res = []
        frame_list = []

        
        for test_file in config['input_data']["dataset"]:
            ls = []
            total_traj = 0
            all_normal_RMSE = []
            device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
            model = net.NetGINConv_ve(num_features, output_size, config).to(device)
            model_folder = 'models/TRAINED/'
            model.load_state_dict(torch.load(os.path.join(config['training']['model_dir'], test_file)+".pt", map_location='cpu'))
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
            if  config['input_data']['subject'] == 'vehicle':
                rmse_batch = torch.full((25,1), 0.0)
                rmse_batch = rmse_batch.squeeze(dim=1)
                rmse_batch = rmse_batch.to(device)
                number_of_traj = 0


            model.eval()
            for batch in tqdm(test_loader):
                batch = [tensor.to(device) for tensor in batch]
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                    non_linear_ped, loss_mask, seq_start_end, frame_id) = batch
                
                if number_of_traj>300:
                    
                    frame_list.append(int(frame_id[0,0,0,0]))


                total_traj += pred_traj_gt.size(0)

                data_list = ut.getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
                graph_batch = tgb.from_data_list(data_list)

                start = time.time()
                # pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
                pred_traj = model.infer(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device), seq_len=test_loader.dataset.pred_len)
                end = time.time()
                if number_of_traj>300:
                    times.append(end-start)

                pred_traj = pred_traj[:, :, :]
                pred_traj = pred_traj.reshape(pred_traj.shape[0],test_loader.dataset.pred_len,2)
                pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

                # pred_traj = pred_traj.reshape(pred_traj.shape[0], config['input_data']['prtediction_step'],config['input_data']['points_per_position']).detach()

                # pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

                ade_batches.append(torch.sum(ut.displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
                fde_batches.append(torch.sum(ut.final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())
        
                if  config['input_data']['subject'] == 'vehicle':
                    rmse_batch += (ut.rmse(pred_traj_real, pred_traj_gt, mode='raw')).detach()
                    number_of_traj+=1
                    all_normal_RMSE.append((ut.rmse(pred_traj_real.to('cpu'), pred_traj_gt.to('cpu'), mode='raw')))
                
                if number_of_traj>5300:
                    break

                # ops, params = profile(model, inputs=(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device)))
                # all_ops.append(ops)
            # all_normal_RMSE = all_normal_RMSE.to('cpu')
        #     array_list = np.array([tensor.numpy() for tensor in all_normal_RMSE])
        #     # Save the list of NumPy arrays to a text file
        #     file_path = '/home/adaneshp/PishguVe_Tran/anomaly_res/All_normal_ADE_FDE_Left_Right_list_data.txt' 
        #     # with open(file_path, 'w') as file:
        #     #     # for array in array_list:
        #     #     np.savetxt(file, array_list, delimiter=',', fmt='%0.3f')
        #     ade = sum(ade_batches) / (total_traj * config['input_data']['prtediction_step'])
        #     fde = sum(fde_batches) / (total_traj)

        #     if  config['input_data']['subject'] == 'vehicle':
        #         rmse = rmse_batch/number_of_traj

        #     print(test_file + "  ADE: " + str(ade) + "  FDE: " + str(fde))
        #     if  config['input_data']['subject'] == 'vehicle':
        #         print("RMSE: 1s,2s,3s,4s,5s")
        #         pred_fde_horiz = ut.horiz_eval(rmse, 5)
        #         print(pred_fde_horiz)
        #     ls.append(test_file)
        #     ls.append(ade)
        #     ls.append(fde)
        #     res.append(ls)
        # avg_fde = 0
        # avg_ade = 0
        # for data_st in res:
        #     avg_ade = avg_ade + data_st[1]
        #     avg_fde = avg_fde + data_st[2]
        # avg_ade = avg_ade/len(res)
        # avg_fde = avg_fde/len(res)



        print("Latency per Trajectory processing " + str(mean(times)) + " sec")

        print("Total Time: " + str(sum(times)) + " sec")

        print ("Total unique Frames ", str(len(set(frame_list))))

        print ("Total no of Vehicles processed or samples per seconds: ", str(total_traj-300))


        
