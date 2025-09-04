import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import pickle
from model_arch import SimpleModel

def make_dir(path):
    os.makedirs(path, exist_ok=True)

def init_model_p_from_three(model_p, m1, m2, m3, fs):
    # zero everything
    for i in [0,2,4,6]:
        model_p.fc[i].weight.data.zero_()
        model_p.fc[i].bias.data.zero_()

    # fc[0]: [3*fs x 39]  <- stack first layers of m1, m2, m3
    model_p.fc[0].weight.data[:fs, :]      = m1.fc[0].weight.data
    model_p.fc[0].weight.data[fs:2*fs, :]  = m2.fc[0].weight.data
    model_p.fc[0].weight.data[2*fs:, :]    = m3.fc[0].weight.data
    model_p.fc[0].bias.data[:fs]           = m1.fc[0].bias.data
    model_p.fc[0].bias.data[fs:2*fs]       = m2.fc[0].bias.data
    model_p.fc[0].bias.data[2*fs:]         = m3.fc[0].bias.data

    # fc[2]: [3*fs x 3*fs]  <- block diagonal of second layers
    model_p.fc[2].weight.data[:fs, :fs]                 = m1.fc[2].weight.data
    model_p.fc[2].weight.data[fs:2*fs, fs:2*fs]         = m2.fc[2].weight.data
    model_p.fc[2].weight.data[2*fs:, 2*fs:]             = m3.fc[2].weight.data
    model_p.fc[2].bias.data[:fs]                        = m1.fc[2].bias.data
    model_p.fc[2].bias.data[fs:2*fs]                    = m2.fc[2].bias.data
    model_p.fc[2].bias.data[2*fs:]                      = m3.fc[2].bias.data

    # fc[4]: [3*64 x 3*fs]  <- block diagonal of third layers
    model_p.fc[4].weight.data[:64, :fs]                 = m1.fc[4].weight.data
    model_p.fc[4].weight.data[64:128, fs:2*fs]          = m2.fc[4].weight.data
    model_p.fc[4].weight.data[128:, 2*fs:]              = m3.fc[4].weight.data
    model_p.fc[4].bias.data[:64]                        = m1.fc[4].bias.data
    model_p.fc[4].bias.data[64:128]                     = m2.fc[4].bias.data
    model_p.fc[4].bias.data[128:]                       = m3.fc[4].bias.data

    # fc[6]: [1 x 3*64]  <- concatenate columns; sum biases
    model_p.fc[6].weight.data[:, :64]                   = m1.fc[6].weight.data
    model_p.fc[6].weight.data[:, 64:128]                = m2.fc[6].weight.data
    model_p.fc[6].weight.data[:, 128:]                  = m3.fc[6].weight.data
    model_p.fc[6].bias.data = (
        m1.fc[6].bias.data + m2.fc[6].bias.data + m3.fc[6].bias.data
    )

    # Copy BN stats (running stats only) from m1 as a baseline
    model_p.batch_norm.running_mean = m1.batch_norm.running_mean.clone()
    model_p.batch_norm.running_var  = m1.batch_norm.running_var.clone()
    model_p.batch_norm.momentum     = 0.0

def build_X(data):
    # data: [N, T, 39]
    X = data.copy()
    # relative re-encoding of first 3 channels
    X[:, :, 0] = data[:, :, 2] - data[:, :, 1]
    X[:, :, 1] = data[:, :, 1] - data[:, :, 0]
    X[:, :, 2] = data[:, :, 2] - data[:, :, 0]

    # angle wrap for first 3 channels into [-75, 75] with period ~150.087
    P = 150.087
    for k in [0, 1, 2]:
        X[:, :, k] = (X[:, :, k] > 75.) * (X[:, :, k] - P) + \
                     (X[:, :, k] < -75.) * (X[:, :, k] + P) + \
                     (X[:, :, k] <= 75.) * (X[:, :, k] >= -75.) * X[:, :, k]
    return X

def make_shifted_prev(X):
    # X: torch [N, T, 39]
    X_prev = X.clone()
    # copy previous time for all but last 15 features (keep last 15 as-is)
    X_prev[1:, :, :-15] = X[:-1, :, :-15]
    return X_prev

def paired_role_slices(X, X_prev, j, batch_size, role_offset):
    """
    Build paired time-aligned slices for a role:
    role_offset: 0 -> v2 (B), 1 -> v3 (C), 2 -> v1 (A).
    We ensure lengths match by using X end=j+bs-1 and X_prev end=j+bs.
    """
    start_X   = j + role_offset
    start_X_p = start_X + 1
    end_X     = j + batch_size - 1  # exclusive
    end_X_p   = j + batch_size      # exclusive

    sX   = X[start_X:end_X:3]
    sX_p = X_prev[start_X_p:end_X_p:3]
    return sX, sX_p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name1', default='none', help='Name of the pre-trained model to represent v1 function')
    parser.add_argument('--model_name2', default='none', help='Name of the pre-trained model to represent v2 function')
    parser.add_argument('--model_name3', default='none', help='Name of the pre-trained model to represent v3 function')
    parser.add_argument('--model_name',  default='none', help='Name of the pre-trained combined potential model_p (if any)')
    parser.add_argument('--data_name',   default='none', help='Dataset name')
    parser.add_argument('--batch_size',  default=241, type=int, help='Batch size (episodes per step)')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--rel',   action='store_true')
    parser.add_argument('--mpc',   action='store_true')
    parser.add_argument('--s',     default=350, type=int, help='Discount horizon S')
    parser.add_argument('--cuda',  action='store_true')
    args = parser.parse_args()

    # Suffix logic
    data_name = args.data_name + ('_mpc' if args.mpc else '')
    suffix = ""
    fs = 128
    if args.small:
        suffix += "_small"
    if args.s < 250:
        suffix += "_myopic"
    if args.s < 50:
        suffix += "_s"
    if args.mpc:
        suffix += "_mpc"

    # Load data
    data_path = f'data/{data_name}.pkl'
    data = np.load(data_path, allow_pickle=True)
    # use smaller subset if requested
    if args.small:
        data = data[:241]
    else:
        data = data[:241*9]
    # data shape expected: [N, T, 39]
    N, T, F = data.shape
    print("Data shape:", data.shape)

    # Build X and shifted X_
    X = build_X(data)
    X = torch.tensor(X, dtype=torch.float32)
    X[torch.isnan(X)] = 0.0
    X_ = make_shifted_prev(X)

    # Device
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    X   = X.to(device)
    X_  = X_.to(device)

    # Models
    model_v1 = SimpleModel(39, [fs, fs, 64], 1)   # v1
    model_v2 = SimpleModel(39, [fs, fs, 64], 1)   # v2
    model_v3 = SimpleModel(39, [fs, fs, 64], 1)   # v3
    model_p  = SimpleModel(39, [3*fs, 3*fs, 3*64], 1)  # combined potential

    # Load v-model weights (respect --rel)
    base_dir = 'q_models_rel' if args.rel else 'q_models'
    m1_path = os.path.join(base_dir, args.model_name1 + suffix + '.pth')
    m2_path = os.path.join(base_dir, args.model_name2 + suffix + '.pth')
    m3_path = os.path.join(base_dir, args.model_name3 + suffix + '.pth')
    print("Loading:", m1_path)
    print("Loading:", m2_path)
    print("Loading:", m3_path)
    model_v1.load_state_dict(torch.load(m1_path, map_location='cpu'))
    model_v2.load_state_dict(torch.load(m2_path, map_location='cpu'))
    model_v3.load_state_dict(torch.load(m3_path, map_location='cpu'))

    # Move v-models and freeze
    model_v1.to(device).eval()
    model_v2.to(device).eval()
    model_v3.to(device).eval()
    for p in list(model_v1.parameters()) + list(model_v2.parameters()) + list(model_v3.parameters()):
        p.requires_grad_(False)

    # Initialize / load model_p
    if args.model_name != 'none':
        model_p.load_state_dict(torch.load(args.model_name + '.pth', map_location='cpu'))
        print(f"Loaded model_p from {args.model_name}.pth")
    else:
        # init as v1+v2+v3
        init_model_p_from_three(model_p, model_v1, model_v2, model_v3, fs)

    model_p.to(device)
    # Optimizers
    learning_rate = 1e-4
    optimizer_fc  = torch.optim.Adam(model_p.fc.parameters(), lr=learning_rate)
    # (there was an optimizer1 for full model; keep only fc params as in your loop)

    # Save initial checkpoint
    out_dir = 'p_models_rel' if args.rel else 'p_models'
    make_dir(out_dir)
    torch.save(model_p.state_dict(), os.path.join(out_dir, 'model_multi' + suffix + '.pth'))

    # Training config
    discount_factor = 0.98
    n_iters = 100000
    batch_size = int(args.batch_size)
    S = args.s
    S_ = T - S
    if S_ <= 0:
        raise ValueError(f"S_ = T - S must be > 0, but got T={T}, S={S} -> S_={S_}. Reduce --s.")

    loss_fn = nn.MSELoss()
    losses1, losses2, losses3 = [], [], []
    max_gt1 = torch.tensor(0.0, device=device)
    max_gt2 = torch.tensor(0.0, device=device)
    max_gt3 = torch.tensor(0.0, device=device)
    min_loss = float('inf')

    model_p.train()  # IMPORTANT: train mode for BN/Dropout

    for i in range(n_iters):
        total_loss = 0.0

        for j in range(0, X.shape[0], batch_size):
            # Ensure the +1 index for X_ stays within range and counts line up
            if j + batch_size > X.shape[0]:
                break

            # Role B (v2): offsets (0, +1)
            Xb, Xb_prev = paired_role_slices(X, X_, j, batch_size, role_offset=0)
            v2   = model_v2(Xb)
            v2_p = model_v2(Xb_prev)
            gt   = (v2 - v2_p).detach()
            preds = model_p(Xb) - model_p(Xb_prev)
            loss  = loss_fn(preds[:, :S_].squeeze(), gt[:, :S_].squeeze())
            optimizer_fc.zero_grad()
            loss.backward()
            if i > 0:
                optimizer_fc.step()

            # Role C (v3): offsets (1, +2)
            Xc, Xc_prev = paired_role_slices(X, X_, j, batch_size, role_offset=1)
            v3   = model_v3(Xc)
            v3_p = model_v3(Xc_prev)
            gt_  = (v3 - v3_p).detach()
            preds_ = model_p(Xc) - model_p(Xc_prev)
            loss_  = loss_fn(preds_[:, :S_].squeeze(), gt_[:, :S_].squeeze())
            optimizer_fc.zero_grad()
            loss_.backward()
            if i > 0:
                optimizer_fc.step()

            # Role A (v1): offsets (2, +3)
            Xa, Xa_prev = paired_role_slices(X, X_, j, batch_size, role_offset=2)
            v1   = model_v1(Xa)
            v1_p = model_v1(Xa_prev)
            gt__ = (v1 - v1_p).detach()
            preds__ = model_p(Xa) - model_p(Xa_prev)
            loss__  = loss_fn(preds__[:, :S_].squeeze(), gt__[:, :S_].squeeze())
            optimizer_fc.zero_grad()
            loss__.backward()
            if i > 0:
                optimizer_fc.step()

            # Accumulate stats
            total_loss += (loss.item() + loss_.item() + loss__.item())

            if i == n_iters - 1:
                losses1.append((preds[:, :S_] - gt[:, :S_]).detach().cpu())
                losses2.append((preds_[:, :S_] - gt_[:, :S_]).detach().cpu())
                losses3.append((preds__[:, :S_] - gt__[:, :S_]).detach().cpu())
                max_gt1 = torch.maximum(max_gt1, torch.max(torch.abs(gt[:, :S_])))
                max_gt2 = torch.maximum(max_gt2, torch.max(torch.abs(gt_[:, :S_])))
                max_gt3 = torch.maximum(max_gt3, torch.max(torch.abs(gt__[:, :S_])))

        print(f"Iteration: {i}  Loss: {total_loss:.6f}")

        if total_loss < min_loss:
            min_loss = total_loss
            torch.save(model_p.state_dict(), os.path.join(out_dir, 'model_multi' + suffix + '.pth'))
            print("Saved model")

    # Save diagnostics
    regret_info = {
        'losses1': losses1,
        'losses2': losses2,
        'losses3': losses3,
        'max_gt1': max_gt1.detach().cpu().item(),
        'max_gt2': max_gt2.detach().cpu().item(),
        'max_gt3': max_gt3.detach().cpu().item(),
    }
    with open('regret_info_multi1' + suffix + '.pkl', 'wb') as f:
        pickle.dump(regret_info, f)

if __name__ == "__main__":
    main()
