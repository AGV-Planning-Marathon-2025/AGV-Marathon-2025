import wandb
import os
import pickle

def merge_artifact_versions(project, entity, artifact_name, version_start, version_end, output_path):
    combined_data = []
    run = wandb.init(project=project, entity=entity, job_type="dataset_merge")

    for v_num in range(version_start, version_end + 1):
        version_tag = f"v{v_num}"
        try:
            artifact_ref = f"{entity}/{project}/{artifact_name}:{version_tag}"
            print(f"Downloading artifact version: {artifact_ref}")
            artifact = run.use_artifact(artifact_ref, type='dataset')
            artifact_dir = artifact.download()
            pkl_path = os.path.join(artifact_dir, 'combined_data.pkl')
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
                combined_data.extend(data)  # Append episodes from this version
            print(f"Loaded {len(data)} episodes from {version_tag}")
        except Exception as e:
            print(f"Warning: Could not load version {version_tag}: {e}")
            # Continue even if some versions are missing or fail

    # Save merged combined data locally
    with open(output_path, 'wb') as f_out:
        pickle.dump(combined_data, f_out)
    print(f"Saved combined dataset with {len(combined_data)} total episodes to {output_path}")

    run.finish()

# Usage example:
if __name__ == "__main__":
    merge_artifact_versions(
        project='alpha-racer-data',
        entity='mahin-sank-iit-kgp',
        artifact_name='alpha_racer_multi_mpc_combined',
        version_start=1,
        version_end=1000,  # or your desired end version
        output_path='combined_full_dataset.pkl'
    )
