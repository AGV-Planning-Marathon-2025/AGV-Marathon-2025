import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import os

# Constants
DEFAULT_BATCH_SIZE = 241
FULL_DATA_MULTIPLIER = 9

def build_suffix(args):   
#add required suffix based on the command line arguments
    suffix = ""
    fs = 128 #fs decides the number of hidden feature layers
    if args.small:
        suffix = "_small"
    if int(args.s) < 250:
        suffix += "_myopic"
    if int(args.s) < 50:
        suffix += "_s"
    if args.mpc:
        suffix += "_mpc"
    return suffix, fs

def zero_initialize_pmodel(model_p):
    #Initializes all weights and biases of the fc layers in model_p to zero.
    for layer in model_p.fc:
        if hasattr(layer, 'weight'):
            nn.init.constant_(layer.weight.data, 0)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)

def build_pmodel_from_submodels(model_p, models, fs):
    """
    Combines the weights and biases from the list of sub-models into model_p
    """
    num_base_models = len(models)
    
    # Layer fc[0]: Simple vertical concatenation of weights and biases
    model_p.fc[0].weight.data = torch.cat([m.fc[0].weight.data for m in models], dim=0)
    model_p.fc[0].bias.data = torch.cat([m.fc[0].bias.data for m in models], dim=0)

    # Layer fc[2]: Create a block-diagonal matrix for weights and biases
    for i in range(num_base_models):
        start, end = i * fs, (i + 1) * fs
        model_p.fc[2].weight.data[start:end, start:end] = models[i].fc[2].weight.data
        model_p.fc[2].bias.data[start:end] = models[i].fc[2].bias.data

    # Layer fc[4]: Create a block-diagonal matrix for weights and biases with different sizes
    for i in range(num_base_models):
        in_start, in_end = i * fs, (i + 1) * fs
        out_start, out_end = i * 64, (i + 1) * 64
        model_p.fc[4].weight.data[out_start:out_end, in_start:in_end] = models[i].fc[4].weight.data
        model_p.fc[4].bias.data[out_start:out_end] = models[i].fc[4].bias.data

    # Layer fc[6]: Horizontal concatenation of weights and summation of biases
    model_p.fc[6].weight.data = torch.cat([m.fc[6].weight.data for m in models], dim=1)
    model_p.fc[6].bias.data = sum([m.fc[6].bias.data for m in models])

    # Copy Batch Normalization parameters
    if hasattr(models[0], 'bn'):
        model_p.bn.running_mean = models[0].bn.running_mean
        model_p.bn.running_var = models[0].bn.running_var
        model_p.bn.momentum = 0.

def initialize_models(args, fs, SimpleModel):
 
    #Initializes the base models and the combined potential model. Create necessary directories if they don't exist
    if not os.path.exists('p_models'):
        os.makedirs('p_models')
    if not os.path.exists('p_models_rel'):
        os.makedirs('p_models_rel')

    # Load base models
    models = []
    model_paths = [
        f"/home/netra/Desktop/trainingp/value_model_0.pth",
        f"/home/netra/Desktop/trainingp/value_model_1.pth",
        f"/home/netra/Desktop/trainingp/value_model_2.pth"
    ]

    for path in model_paths:
        model = SimpleModel(39, [fs, fs, 64], 1)
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')), strict=False)
        model.eval()
        models.append(model)

    # Initialize the combined potential model
    num_base_models = len(models)
    model_p = SimpleModel(39, [num_base_models*fs, num_base_models*fs, num_base_models*64], 1)

    # If no pre-trained potential model is provided, build it from the base models
    if args.model_name == 'none':
        zero_initialize_pmodel(model_p)
        build_pmodel_from_submodels(model_p, models, fs)
    else:
        # Load a pre-trained potential model if specified
        model_p.load_state_dict(torch.load(args.model_name, map_location=torch.device('cpu')))

    model_p.eval()
    return {"models": models, "model_p": model_p}


def preprocess_data(data, cuda=False):
    """
    Preprocesses the raw data by computing relative differences and converting to tensors.
    """
    X = data.copy()

    # Compute relative differences and handle cyclic values
    X[:, :, 0] = X[:, :, 2] - X[:, :, 1]
    X[:, :, 1] -= data[:, :, 0]
    X[:, :, 2] -= data[:, :, 0]
    X[:, :, :3] = ((X[:, :, :3] > 75) * (X[:, :, :3] - 150.087) +
                   (X[:, :, :3] < -75) * (X[:, :, :3] + 150.087) +
                   ((X[:, :, :3] >= -75) & (X[:, :, :3] <= 75)) * X[:, :, :3])

    # Convert to PyTorch tensor and handle NaNs
    X = torch.from_numpy(X).float()
    X = torch.nan_to_num(X, nan=0.0)

    # Move to GPU if requested
    if cuda and torch.cuda.is_available():
        X = X.cuda()

    # Create the shifted tensor for temporal differences
    X_shift = X.clone()
    X_shift[1:, :, :-15] = X[:-1, :, :-15].clone()

    return X, X_shift

def train_model(args, X, X_shift, model_p, models):
    """
    Performs the training loop for the potential model.
    """
    # Set potential model to training mode
    model_p.train()

    S = int(args.s)
    S_ = 500 - S
    n_iters = 100000
    batch_size = int(args.batch_size)
    learning_rate = 0.0001

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model_p.fc.parameters(), lr=learning_rate)

    min_loss = float('inf')

    # Storage for final losses for analysis
    losses1, losses2, losses3 = [], [], []
    max_gt1, max_gt2, max_gt3 = 0., 0., 0.

    print("Starting training loop...")
    for i in range(n_iters):
        total_loss = 0.0

        # Iterate over batches
        for j in range(0, X.shape[0], batch_size):
            if j + batch_size >= X.shape[0]:
                break

            # Correct batch slicing from the original script
            X_batch1 = X[j + 2:j + batch_size - 1:3].reshape(-1, 39)
            X_batch2 = X[j:j + batch_size - 1:3].reshape(-1, 39)
            X_batch3 = X[j + 1:j + batch_size - 1:3].reshape(-1, 39)

            X_batch1_ = X_shift[j + 3:j + batch_size:3].reshape(-1, 39)
            X_batch2_ = X_shift[j + 1:j + batch_size:3].reshape(-1, 39)
            X_batch3_ = X_shift[j + 2:j + batch_size:3].reshape(-1, 39)

            # Forward pass through base models (detach to prevent gradients)
            with torch.no_grad():
                v1 = models[0](X_batch1)
                v2 = models[1](X_batch2)
                v3 = models[2](X_batch3)

                v1_ = models[0](X_batch1_)
                v2_ = models[1](X_batch2_)
                v3_ = models[2](X_batch3_)

            # Compute ground truths
            gt = v2 - v2_
            gt_ = v3 - v3_
            gt__ = v1 - v1_

            # Forward pass through potential model
            preds = model_p(X_batch2) - model_p(X_batch2_)
            preds_ = model_p(X_batch3) - model_p(X_batch3_)
            preds__ = model_p(X_batch1) - model_p(X_batch1_)

            # Compute and combine losses
            loss = loss_fn(preds[:, :S_].squeeze(), gt[:, :S_].squeeze().float())
            loss_ = loss_fn(preds_[:, :S_].squeeze(), gt_[:, :S_].squeeze().float())
            loss__ = loss_fn(preds__[:, :S_].squeeze(), gt__[:, :S_].squeeze().float())

            total_loss_combined = loss + loss_ + loss__

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss_combined.backward()
            optimizer.step()

            total_loss += total_loss_combined.item()

            # Store final losses for analysis
            if i == n_iters - 1:
                losses1.append((preds[:, :S_].squeeze() - gt[:, :S_].squeeze()).detach().cpu())
                max_gt1 = max(max_gt1, torch.max(torch.abs(gt[:, :S_])).item())

                losses2.append((preds_[:, :S_].squeeze() - gt_[:, :S_].squeeze()).detach().cpu())
                max_gt2 = max(max_gt2, torch.max(torch.abs(gt_[:, :S_])).item())

                losses3.append((preds__[:, :S_].squeeze() - gt__[:, :S_].squeeze()).detach().cpu())
                max_gt3 = max(max_gt3, torch.max(torch.abs(gt__[:, :S_])).item())

        print(f"Iteration: {i}, Loss: {total_loss}")

        # Save the best model
        if total_loss < min_loss:
            min_loss = total_loss
            save_path = f'p_models_rel/model_multi{suffix}.pth' if args.rel else f'p_models/model_multi{suffix}.pth'
            torch.save(model_p.state_dict(), save_path)
            print("Saved best model.")

    # Save final regret info
    regret_info = {
        'losses1': losses1, 'losses2': losses2, 'losses3': losses3,
        'max_gt1': max_gt1, 'max_gt2': max_gt2, 'max_gt3': max_gt3
    }
    with open(f'regret_info_multi1{suffix}.pkl', 'wb') as f:
        pickle.dump(regret_info, f)

    print("Training complete.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='none', help='Name of the pre-trained model to represent potential function')
    parser.add_argument('--data_name', default='none', help='Name of the previous dataset where potential function could be evaluated to check where it is failing')
    parser.add_argument('--batch_size', default=241, type=int, help='Batch size')
    parser.add_argument('--small', action='store_true', help='Whether to train on small dataset')
    parser.add_argument('--rel', action='store_true', help='Whether to train on rel q models')
    parser.add_argument('--mpc', action='store_true', help='Whether to run on MPC collected dataset')
    parser.add_argument('--s', default=350, type=int, help='Discount factor')
    parser.add_argument('--cuda', action='store_true', help='Whether to use cuda or not')
    args = parser.parse_args()

    try:
        from model_arch import SimpleModel
    except ImportError:
        raise ImportError("Please ensure model_arch.py is in the same directory.")

    # Data loading
    if args.mpc:
        args.data_name += "_mpc"
    data_multiplier = 1 if args.small else FULL_DATA_MULTIPLIER

    # Original data path
    data_path = '/home/netra/Desktop/trainingp/data.pkl'
    try:
        data = np.load(data_path, allow_pickle=True)
        data = data[:DEFAULT_BATCH_SIZE * data_multiplier]
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    # Build suffix and feature size
    suffix, fs = build_suffix(args)

    # Initialize all models
    models_dict = initialize_models(args, fs, SimpleModel)
    models = models_dict["models"]
    model_p = models_dict["model_p"]

    # Check for CUDA availability and move models
    if args.cuda and torch.cuda.is_available():
        print("Using CUDA.")
        for m in models:
            m.cuda()
        model_p.cuda()
    else:
        print("Using CPU.")

    # Preprocess the entire dataset once, before the training loop
    X, X_shift = preprocess_data(data, cuda=args.cuda)

    # Train the potential model
    train_model(args, X, X_shift, model_p, models)

    print("Script finished successfully.")
