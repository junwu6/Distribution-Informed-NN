from __future__ import print_function
import argparse
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from DINO import DINO
import torch.optim as optim
from neural_tangents import stax
import neural_tangents as nt

# Command setting
parser = argparse.ArgumentParser(description='Domain Adaptation Regression')
parser.add_argument('-cuda', type=int, default=1, help='cuda id')
parser.add_argument('-root_dir', type=str, default='../data/')
parser.add_argument('-dataset', type=str, default='dSprites')
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-lr', type=float, default=0.1)
parser.add_argument('-weight_decay', type=float, default=5e-4)
args = parser.parse_args()


def train(kernel_fn, s_train_x, s_train_y, t_train_x, t_train_y, t_train_x_unlab, t_test_x, t_test_y, device):
    NUM_BASE = 5000
    k_s_base = np.asarray(kernel_fn(s_train_x[:NUM_BASE], None, 'nngp'))
    k_t_base = np.asarray(kernel_fn(t_train_x_unlab[:NUM_BASE], None, 'nngp'))
    k_st_base = np.asarray(kernel_fn(s_train_x[:NUM_BASE], t_train_x_unlab[:NUM_BASE], 'nngp'))
    model = DINN(device=device, dim=s_train_x.shape[1], base_s=s_train_x[:NUM_BASE], base_t=t_train_x_unlab[:NUM_BASE],
                 nngp_kernels=[k_s_base, k_t_base, k_st_base]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    result = []
    for epoch in range(1, args.epochs+1):
        model.train()
        idx = np.random.choice(s_train_x.shape[0], 500, replace=False)
        source_x = torch.tensor(s_train_x[idx], requires_grad=False, dtype=torch.float).to(device)
        source_y = torch.tensor(s_train_y[idx], requires_grad=False, dtype=torch.float).to(device)
        target_x = torch.tensor(t_train_x, requires_grad=False, dtype=torch.float).to(device)
        target_y = torch.tensor(t_train_y, requires_grad=False, dtype=torch.float).to(device)

        k_ss = torch.tensor(np.asarray(kernel_fn(s_train_x[idx], None, 'nngp')), requires_grad=False, dtype=torch.float).to(device)
        k_tt = torch.tensor(np.asarray(kernel_fn(t_train_x, None, 'nngp')), requires_grad=False, dtype=torch.float).to(device)
        k_st = torch.tensor(np.asarray(kernel_fn(s_train_x[idx], t_train_x, 'nngp')), requires_grad=False, dtype=torch.float).to(device)

        optimizer.zero_grad()
        loss = model(source_x, source_y, target_x, target_y, k_ss, k_tt, k_st)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_x = torch.tensor(t_test_x, requires_grad=False, dtype=torch.float).to(device)
        k_s_ts = torch.tensor(np.asarray(kernel_fn(s_train_x[idx], t_test_x, 'nngp')), requires_grad=False, dtype=torch.float).to(device)
        k_t_ts = torch.tensor(np.asarray(kernel_fn(t_train_x, t_test_x, 'nngp')), requires_grad=False, dtype=torch.float).to(device)
        preds = model.inference(test_x, source_x, source_y, target_x, target_y, k_ss, k_tt, k_st, k_s_ts, k_t_ts)
        mae = mean_absolute_error(t_test_y, preds.detach().cpu().numpy())
        result.append(mae)
        print('MAE: {:.6f}, Loss: {:.6f}'.format(mae, loss.item()))
    return mae


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')

    for dataset in ['dSprites']:
        if dataset == 'dSprites':
            output_dim = 3
            domains = ['color', 'noisy', 'scream']
        elif dataset == 'MPI3D':
            output_dim = 2
            domains = ['real', 'realistic', 'toy']

        init_fn, apply_fn, kernel_fn = stax.serial(
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(1024, W_std=1.5, b_std=0.05),
            stax.Relu(),
            stax.Dense(output_dim, W_std=1.5, b_std=0.05)
        )
        kernel_fn = nt.batch(kernel_fn, device_count=1, batch_size=1000)

        for source in domains:
            for target in domains:
                if source == target:
                    continue
                print("Source: {} -> Target: {}".format(source, target))
                s_data = pickle.load(open("../data/save/{}.pkl".format(source), "rb"))
                t_data = pickle.load(open("../data/save/{}.pkl".format(target), "rb"))
                t_data_test = pickle.load(open("../data/save/{}_test.pkl".format(target), "rb"))

                s_train_x = s_data['X'].astype(float)
                s_train_y = s_data['Y'].astype(float)
                t_train_x = t_data['X'][:100].astype(float)
                t_train_y = t_data['Y'][:100].astype(float)
                t_train_x_unlab = t_data['X'][100:].astype(float)
                t_test_x = t_data_test['X'].astype(float)
                t_test_y = t_data_test['Y'].astype(float)
                if dataset == 'MPI3D':
                    s_train_y = s_train_y / 39
                    t_train_y = t_train_y / 39
                    t_test_y = t_test_y / 39

                test_acc = train(kernel_fn, s_train_x, s_train_y, t_train_x, t_train_y, t_train_x_unlab, t_test_x, t_test_y, device)
                print('{}Test acc: {}'.format('*' * 100, test_acc))
