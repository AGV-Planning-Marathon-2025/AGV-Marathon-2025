import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import argparse
import os
import random
import wandb  # Import the wandb library
from model_arch import SimpleModel

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model_name1', default='none', help='Name of the pre-trained model to represent v1 function')
parser.add_argument('--model_name2', default='none', help='Name of the pre-trained model to represent v2 function')
parser.add_argument('--model_name3', default='none', help='Name of the pre-trained model to represent v3 function')
parser.add_argument('--model_name', default='none', help='Name of the pre-trained model to represent potential function')
parser.add_argument('--data_name', default='none', help='Name of the previous dataset where potential function could be evaluated to check where it is failing')
parser.add_argument('--batch_size', default=241, type=int, help='Batch size')
parser.add_argument('--small', action='store_true', help='Whether to train on small dataset')
parser.add_argument('--rel', action='store_true', help='Whether to train on rel q models')
parser.add_argument('--mpc', action='store_true', help='Whether to run on MPC collected dataset')
parser.add_argument('--s', default=350, type=int, help='Discount factor')
parser.add_argument('--cuda', action='store_true', help='Whether to use cuda or not')
args = parser.parse_args()

# Initialize a new W&B run
# Replace 'my-team-workspace' with your actual team name.
# It will automatically be created if it doesn't exist.
wandb.init(project="p-model-training", entity="mahin-sank-iit-kgp", config=vars(args))
wandb.run.name = f"run_{wandb.run.id}"  # Give the run a unique name

# Update data name based on mpc argument
if args.mpc:
    args.data_name += '_mpc'

# Create the directories if they don't exist
if not os.path.exists('p_models'):
    os.makedirs('p_models')
if not os.path.exists('p_models_rel'):
    os.makedirs('p_models_rel')

# Load data based on small argument
if args.small:
    data = np.load('/home/netra/Desktop/trainingp/data.pkl', allow_pickle=True)[:241]
else:
    data = np.load('/home/netra/Desktop/trainingp/data.pkl', allow_pickle=True)[:241*9]

S = args.s
suffix = ""
fs = 128

# Set suffix and fs based on command-line arguments
if args.small:
    suffix = "_small"
    fs = 128
if S < 250:
    suffix += "_myopic"
    fs = 128
if S < 50:
    suffix += "_s"
    fs = 128
if args.mpc:
    suffix += "_mpc"

discount_factor = 0.98
n_iters = 1000
n_iters_out = 0
learning_rate = 0.0001

# Initialize models
model = SimpleModel(39, [fs, fs, 64], 1)
model_ = SimpleModel(39, [fs, fs, 64], 1)
model__ = SimpleModel(39, [fs, fs, 64], 1)
model_p = SimpleModel(39, [3*fs, 3*fs, 3*64], 1)

# Load pre-trained model weights for the base models
# Use map_location to ensure models are loaded onto the CPU
model.load_state_dict(torch.load('/home/netra/Desktop/trainingp/value_model_0.pth', map_location=torch.device('cpu')), strict=False)
model_.load_state_dict(torch.load('/home/netra/Desktop/trainingp/value_model_1.pth', map_location=torch.device('cpu')), strict=False)
model__.load_state_dict(torch.load('/home/netra/Desktop/trainingp/value_model_2.pth', map_location=torch.device('cpu')), strict=False)

# Set models to evaluation mode
model.eval()
model_.eval()
model__.eval()

# Initialize model_p weights if model_name is not provided
if args.model_name == 'none':
    # Intialize model_p with 0 weights
    model_p.fc[0].weight.data *= 0
    model_p.fc[0].bias.data *= 0
    model_p.fc[2].weight.data *= 0
    model_p.fc[2].bias.data *= 0
    model_p.fc[4].weight.data *= 0
    model_p.fc[4].bias.data *= 0
    model_p.fc[6].weight.data *= 0
    model_p.fc[6].bias.data *= 0

    # Initialize model_p such that model_p(x) = model(x) + model_(x) + model__(x)
    model_p.fc[0].weight.data[:fs, :] = model.fc[0].weight.data
    model_p.fc[0].weight.data[fs:2*fs, :] = model_.fc[0].weight.data
    model_p.fc[0].weight.data[2*fs:, :] = model__.fc[0].weight.data
    model_p.fc[0].bias.data[:fs] = model.fc[0].bias.data
    model_p.fc[0].bias.data[fs:2*fs] = model_.fc[0].bias.data
    model_p.fc[0].bias.data[2*fs:] = model__.fc[0].bias.data

    model_p.fc[2].weight.data[:fs, :fs] = model.fc[2].weight.data
    model_p.fc[2].weight.data[fs:2*fs, fs:2*fs] = model_.fc[2].weight.data
    model_p.fc[2].weight.data[2*fs:, 2*fs:] = model__.fc[2].weight.data
    model_p.fc[2].bias.data[:fs] = model.fc[2].bias.data
    model_p.fc[2].bias.data[fs:2*fs] = model_.fc[2].bias.data
    model_p.fc[2].bias.data[2*fs:] = model__.fc[2].bias.data

    model_p.fc[4].weight.data[:64, :fs] = model.fc[4].weight.data
    model_p.fc[4].weight.data[64:2*64, fs:2*fs] = model_.fc[4].weight.data
    model_p.fc[4].weight.data[2*64:, 2*fs:] = model__.fc[4].weight.data
    model_p.fc[4].bias.data[:64] = model.fc[4].bias.data
    model_p.fc[4].bias.data[64:2*64] = model_.fc[4].bias.data
    model_p.fc[4].bias.data[2*64:] = model__.fc[4].bias.data

    model_p.fc[6].weight.data[:, :64] = model.fc[6].weight.data
    model_p.fc[6].weight.data[:, 64:2*64] = model_.fc[6].weight.data
    model_p.fc[6].weight.data[:, 2*64:] = model__.fc[6].weight.data
    model_p.fc[6].bias.data = (
        model.fc[6].bias.data + model_.fc[6].bias.data + model__.fc[6].bias.data
    )

    # Correcting attribute name from batch_norm to bn
    model_p.bn.running_mean = model.bn.running_mean
    model_p.bn.running_var = model.bn.running_var
    model_p.bn.momentum = 0.0

if args.rel:
    model.load_state_dict(torch.load(args.model_name1, map_location=torch.device('cpu')), strict=False)
    model_.load_state_dict(torch.load(args.model_name2, map_location=torch.device('cpu')), strict=False)
    model__.load_state_dict(torch.load(args.model_name3, map_location=torch.device('cpu')), strict=False)

# Print model information
print(args.model_name1 + suffix + '.pth')
print(model_p.fc[0].weight.data[:fs, :])
print(model.fc[0].weight.data)
model.eval()
model_.eval()
model__.eval()

# Example tensor initialization
X_eg = torch.tensor([
    [
        6.2830, 5.7749, 12.0579, -0.3827, -0.3814, -0.0952, 0.0330, 1.4586,
        -0.0798, -0.4122, -0.0144, 1.4569, -0.0832, -0.4455, 0.2935, 1.4543,
        0.0778, 0.5303, 0.0260, 0.0270, 0.0533, 0.0273, 0.0262, 0.0536,
        0.1000, 0.4071, 0.1532, 0.8500, 0.0, 0.1000, 0.1525, 0.3020,
        1.0000, 0.7273, 0.2374, 0.4838, 0.2773, 1.0545, 0.7273
    ]
] * 100)
print(X_eg.shape)
X_eg[:, -12] = torch.tensor(np.linspace(0.85, 1.1, 100))

# The cuda lines were commented out previously to run on CPU
# if args.cuda:
#     model = model.cuda()
#     model_ = model.cuda()
#     model__ = model__.cuda()
#     model_p = model_p.cuda()
#     X_eg = X_eg.cuda()

# Loss function and optimizer initialization
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_p.fc.parameters(), lr=learning_rate)

# Data preprocessing
X = data[:, :, :].copy()
X[:, :, 0] = data[:, :, 2] - data[:, :, 1]
X[:, :, 1] -= data[:, :, 0]
X[:, :, 2] -= data[:, :, 0]
X[:, :, 0] = (
    (X[:, :, 0] > 75.0) * (X[:, :, 0] - 150.087)
    + (X[:, :, 0] < -75.0) * (X[:, :, 0] + 150.087)
    + (X[:, :, 0] <= 75.0) * (X[:, :, 0] >= -75.0) * X[:, :, 0]
)
X[:, :, 1] = (
    (X[:, :, 1] > 75.0) * (X[:, :, 1] - 150.087)
    + (X[:, :, 1] < -75.0) * (X[:, :, 1] + 150.087)
    + (X[:, :, 1] <= 75.0) * (X[:, :, 1] >= -75.0) * X[:, :, 1]
)
X[:, :, 2] = (
    (X[:, :, 2] > 75.0) * (X[:, :, 2] - 150.087)
    + (X[:, :, 2] < -75.0) * (X[:, :, 2] + 150.087)
    + (X[:, :, 2] <= 75.0) * (X[:, :, 2] >= -75.0) * X[:, :, 2]
)

X = torch.from_numpy(X).float()
X[torch.isnan(X)] = 0.0

X_ = X.clone()
X_[1:, :, :-15] = X[:-1, :, :-15]

# Print shapes for debugging
print(X.shape)
print(X[:5, 100, -15:])

# Initial phase
model_p.eval()
S_ = 500 - S
losses1 = []
losses2 = []
losses3 = []
max_gt1 = 0.0
max_gt2 = 0.0
max_gt3 = 0.0
min_loss = 334.0

# Training loop
for i in range(n_iters):
    total_loss = 0.0
    for j in range(0, X.shape[0], args.batch_size):
        if j + args.batch_size > X.shape[0]:
            break

        # Extract batches and reshape for the models
        X_batch1 = X[j + 2:j + args.batch_size - 1:3].reshape(-1, 39)
        X_batch2 = X[j:j + args.batch_size - 1:3].reshape(-1, 39)
        X_batch3 = X[j + 1:j + args.batch_size - 1:3].reshape(-1, 39)

        X_batch1_ = X_[j + 3:j + args.batch_size:3].reshape(-1, 39)
        X_batch2_ = X_[j + 1:j + args.batch_size:3].reshape(-1, 39)
        X_batch3_ = X_[j + 2:j + args.batch_size:3].reshape(-1, 39)

        # Pass reshaped tensors to models
        v1 = model(X_batch1)
        v2 = model_(X_batch2)
        v3 = model__(X_batch3)

        v1_ = model(X_batch1_)
        v2_ = model_(X_batch2_)
        v3_ = model__(X_batch3_)

        # Calculate loss
        gt = v2 - v2_
        gt_ = v3 - v3_
        gt__ = v1 - v1_

        preds = model_p(X_batch2) - model_p(X_batch2_)
        preds_ = model_p(X_batch3) - model_p(X_batch3_)
        preds__ = model_p(X_batch1) - model_p(X_batch1_)

        # Combine losses and perform a single backward pass
        loss = loss_fn(preds[:, :S_].squeeze(), gt[:, :S_].squeeze().float())
        loss_ = loss_fn(preds_[:, :S_].squeeze(), gt_[:, :S_].squeeze().float())
        loss__ = loss_fn(preds__[:, :S_].squeeze(), gt__[:, :S_].squeeze().float())

        total_loss_combined = loss + loss_ + loss__

        optimizer.zero_grad()
        total_loss_combined.backward()

        if i > 0:
            optimizer.step()

        # Append losses for analysis
        if i == n_iters - 1:
            losses1.append(preds[:, :S_].squeeze() - gt[:, :S_].squeeze())
            max_gt1 = max(max_gt1, torch.max(torch.abs(gt[:, :S_])))

            losses2.append(preds_[:, :S_].squeeze() - gt_[:, :S_].squeeze())
            max_gt2 = max(max_gt2, torch.max(torch.abs(gt_[:, :S_])))

            losses3.append(preds__[:, :S_].squeeze() - gt__[:, :S_].squeeze())
            max_gt3 = max(max_gt3, torch.max(torch.abs(gt__[:, :S_])))

        total_loss += total_loss_combined.item()

    # Log the total loss to W&B
    wandb.log({"total_loss": total_loss}, step=i)

    print("Iteration: ", i, " Loss: ", total_loss)
    if total_loss < min_loss:
        min_loss = total_loss
        if args.rel:
            torch.save(model_p.state_dict(), 'p_models_rel/model_multi' + suffix + '.pth')
        else:
            torch.save(model_p.state_dict(), 'p_models/model_multi' + suffix + '.pth')
        print("Saved model")
        # Create an artifact and add the model file to it
        model_artifact = wandb.Artifact(
            name=f"p_model_multi{suffix}",
            type="model",
            description="The trained p-model with the best validation loss."
        )
        model_artifact.add_file('/home/netra/Desktop/trainingp/p_models/model_multi.pth')

        # Log the artifact to your W&B run
        wandb.log_artifact(model_artifact)

print("Saved and logged best model.")

regret_info = {
    'losses1': losses1,
    'losses2': losses2,
    'losses3': losses3,
    'max_gt1': max_gt1,
    'max_gt2': max_gt2,
    'max_gt3': max_gt3
}
with open('regret_info_multi1' + suffix + '.pkl', 'wb') as f:
    pickle.dump(regret_info, f)

# Finish the W&B run
wandb.finish()
