import wandb

# Start a run under your team + project
wandb.init(
    entity="mahin-sank-iit-kgp",   # your team name
    project="recorded-races",      # your project name
    name="upload-racedata"         # optional run name
)

# Create an artifact
artifact = wandb.Artifact(
    name="racedata",   # logical name inside W&B
    type="dataset"     # can be "dataset", "model", etc.
)

# Add your file
artifact.add_file("../car_ros2/car_ros2/recorded_races/racedata_ours_vs_mpc_vs_mpc_grad.pkl")

# Log (upload) it
wandb.log_artifact(artifact)

# Finish run
wandb.finish()