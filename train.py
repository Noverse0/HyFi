import wandb
import time
import math

import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
from model.loader import DatasetLoader
from model.models import FG_HGCL, HGNN
from model.evaluation import linear_evaluation
from model.utils import fix_seed, plot_tsne, cluster_eval

def train(data, model, optimizer, params, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    features = data.features
    hyperedge_index = data.hyperedge_index
    dropout_rate = params['dropout_rate']
    noise_std_n = params['noise_std_n']
    noise_std_e = params['noise_std_e']

    # Encoder
    n1, e1 = model(features, hyperedge_index, dropout_rate)
    n1, e1 = model.node_projection(n1), model.edge_projection(e1)
    
    # make multiple views
    n2 = torch.zeros((params['num_views'], n1.shape[0], n1.shape[1])).to(args.device)
    e2 = torch.zeros((params['num_views'], e1.shape[0], e1.shape[1])).to(args.device)
    for i in range(params['num_views']):
        n3, e3 = model(features, hyperedge_index, dropout_rate, noise_std_n, params['noise_std_e'])
        n2[i] = model.node_projection(n3)
        e2[i] = model.edge_projection(e3)
    
    loss_n = model.calulate_loss(n1, n2, data.overlab_hyperedge, params['tau_n'], batch_size=params['batch_size1'])
    loss_g = model.calulate_loss(e1, e2, data.overlab_hypernode, params['tau_g'], batch_size=params['batch_size2'])
    
    loss = loss_n + params['w_g'] * loss_g
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), model

def node_classification_eval(model, data, params, seed, num_splits=20):
    model.eval()
    n, _ = model(data.features, data.hyperedge_index)
    
    lr = params['eval_lr']
    max_epoch = params['eval_epochs']
    
    accs = []
    for i in range(num_splits):
        masks = data.generate_random_split(seed=i)
        accs.append(linear_evaluation(n, data.labels, masks, lr=lr, max_epoch=max_epoch))
        
    # nmi, f1
    n = n.cpu().detach().numpy()
    labels = data.labels.cpu().detach().numpy()
    nmi, f1 = cluster_eval(n, labels)
    # plot_tsne(n, data.labels, dir="tsne_figure", file_name=f'{args.dataset}_{seed}_fghgcl.pdf')
            
    return accs, nmi, f1


def main():
    wandb.init(project="hyfi_ablation", mode="online")
    params = yaml.safe_load(open('config.yaml'))[args.dataset]
    tuning_params = wandb.config
    params.update(tuning_params)
    
    start = time.time()
    math.factorial(100000)
    
    print(params)
    data = DatasetLoader().load(args.dataset).to(args.device)
    accs ,nmis, f1s = [], [], []
    gpu_memory_allocated = []
    gpu_max_memory_allocated = []
    for seed in range(args.num_seeds):
        # Reset the peak memory stats at the beginning of each iteration
        torch.cuda.reset_peak_memory_stats(args.device)
        
        fix_seed(seed)
        encoder = HGNN(data.features.shape[1], params['hid_dim'], params['hid_dim'], params['num_layers'])
        model = FG_HGCL(encoder, params['proj_dim'], args.device).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        wandb.watch(model, log='all')
        for epoch in tqdm(range(1, params['epochs'] + 1)):
            loss, model = train(data, model, optimizer, params, epoch)
            wandb.log({"Epoch": epoch, "Loss": loss})
            
            # if epoch < 51:
            #     acc, nmi, f1 = node_classification_eval(model, data, params, seed)
            #     acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
            #     wandb.log({"Epoch": epoch,  "Loss": loss, "ACC": acc_mean[2], "NMI": nmi, "F1": f1})
        
        # At the end of each seed iteration, record the memory usage
        gpu_memory_allocated.append(torch.cuda.memory_allocated(args.device))
        gpu_max_memory_allocated.append(torch.cuda.max_memory_allocated(args.device))
        
        # evaluation
        acc, nmi, f1 = node_classification_eval(model, data, params, seed)
        accs.append(acc)
        nmis.append(nmi)
        f1s.append(f1)
        acc_mean, acc_std = np.mean(acc, axis=0), np.std(acc, axis=0)
        print(f'seed: {seed}, train_acc: {acc_mean[0]:.2f}±{acc_std[0]:.2f}, '
            f'valid_acc: {acc_mean[1]:.2f}±{acc_std[1]:.2f}, test_acc: {acc_mean[2]:.2f}±{acc_std[2]:.2f}')
        print(f'Clustering NMI: {nmi:.2f}, F1: {f1:.2f}')
        
    end = time.time()
    print(f"{(end - start)/5:.2f} sec")
    
    accs = np.array(accs).reshape(-1, 3)
    accs_mean = list(np.mean(accs, axis=0))
    accs_std = list(np.std(accs, axis=0))
    print(f'[Final] dataset: {args.dataset}')
    print(f'test_acc: {accs_mean[2]:.2f}±{accs_std[2]:.2f}')
    print(f'Clustering NMI: {np.mean(nmis, axis=0):.2f}, F1: {np.mean(f1s, axis=0):.2f}')
    
    # Calculate and print the average GPU memory usage
    avg_gpu_memory_allocated = sum(gpu_memory_allocated) / len(gpu_memory_allocated)
    avg_gpu_max_memory_allocated = sum(gpu_max_memory_allocated) / len(gpu_max_memory_allocated)
    print(f'Average allocated GPU memory: {avg_gpu_memory_allocated / (1024 ** 2):.2f} MB')
    print(f'Average peak allocated GPU memory: {avg_gpu_max_memory_allocated / (1024 ** 2):.2f} MB')    
    
    wandb.log({
        "dataset": args.dataset,
        "Train_Acc_Mean": accs_mean[0],
        "Train_Acc_Std": accs_std[0],
        "Valid_Acc_Mean": accs_mean[1],
        "Valid_Acc_Std": accs_std[1],
        "Test_Acc_Mean": accs_mean[2],
        "Test_Acc_Std": accs_std[2],
        "NMI_Mean": np.mean(nmis, axis=0),
        "F1_Mean": np.mean(f1s, axis=0),
        "Time": (end - start)/5,
        "Average_GPU_Memory_Allocated_MB": avg_gpu_memory_allocated / (1024 ** 2),
        "Average_GPU_Peak_Memory_Allocated_MB": avg_gpu_max_memory_allocated / (1024 ** 2),
        "model": 'HyFi',
        "ablation": "w/o_edge_loss",
        "gpu": "A6000"
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HyFi unsupervised learning.')
    parser.add_argument('--dataset', type=str, default='cora', 
        choices=['cora', 'citeseer', 'pubmed', 'cora_coauthor', 'dblp_coauthor', 
                 'zoo', '20newsW100', 'Mushroom', 'NTU2012', 'ModelNet40'])
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    # main()
    
    sweep_configuration = {
    "name": args.dataset,
    "method": "grid",
    "metric": {"goal": "maximize", "name": "Test_Acc_Mean"},
    "parameters": {
        "num_layers": {"values": [1]},
        # "hid_dim": {"values": [128, 256, 512, 1024]},
        # "proj_dim": {"values": [128, 256, 512, 1024]},
        # "lr": {"values": [1.0e-03, 1.0e-04, 5.0e-03, 5.0e-04]},
        # "epochs": {"values": [150, 200, 300]},
        # "weight_decay": {"values": [1.0e-04, 1.0e-05]},
        # "tau_n": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        # "tau_g": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
        "w_g": {"values": [0.0]},
        # "eval_lr": {"values": [0.0001, 0.005, 0.001, 0.05, 0.01]},
        # "eval_epochs": {"values": [50, 100, 150, 200, 300]},
        # "dropout_rate": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        # "noise_std_n": {"values": [0.0]},
        # "noise_std_n": {"values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        # "noise_std_e": {"values": [0.0, 0.1, 0.01, 0.001, 0.0001]},
        # "batch_size1": {"values": [None]},
        # "batch_size2": {"values": [None]},
        # "num_views": {"values": [1,2,3,4,5]},
        # "num_views": {"values": [0]},
        },
    }
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="hyfi_ablation")
    wandb.agent(sweep_id, main)
    wandb.finish()
    