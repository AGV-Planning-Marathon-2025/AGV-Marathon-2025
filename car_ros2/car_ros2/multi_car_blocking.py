import os
import time
import argparse
import pickle
import io
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point, PolygonStamped, Point32, TwistStamped
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np

from tf_transformations import quaternion_from_euler, euler_matrix

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from std_msgs.msg import Float64, Int8
import torch
import jax
import jax.numpy as jnp


class WandbArtifactManager:
    def __init__(self, project, entity, artifact_name, max_retries=3, backoff_base=2.0):
        self.project = project
        self.entity = entity
        self.artifact_name = artifact_name
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.run = None
        self.data_buffer = []
        self._init_run_and_load_existing()

    def _init_run_and_load_existing(self):
        self.run = wandb.init(project=self.project, entity=self.entity, name=f"{self.artifact_name}_run", reinit=True)
        try:
            print(f"[W&B] Loading existing artifact: {self.artifact_name}:latest")
            self.artifact = self.run.use_artifact(f"{self.project}/{self.artifact_name}:latest", type='dataset')
            artifact_dir = self.artifact.download(replace=True, root="tmp_wandb")
            artifact_file_path = os.path.join(artifact_dir, "combined_data.pkl")
            if os.path.exists(artifact_file_path):
                with open(artifact_file_path, 'rb') as f:
                    self.data_buffer = pickle.load(f)
                print(f"[W&B] Loaded combined data with {len(self.data_buffer)} previous episodes")
            else:
                print("[W&B] combined_data.pkl not found, starting new.")
                self.data_buffer = []
        except Exception as e:
            print(f"[W&B] No existing artifact or error: {e}")
            self.data_buffer = []

    def append_episode(self, data):
        self.data_buffer.append(data)
        print(f"[W&B] Appended episode data, total episodes: {len(self.data_buffer)}")

    def upload_combined(self):
        attempts = 0
        while attempts < self.max_retries:
            try:
                artifact = wandb.Artifact(name=self.artifact_name, type='dataset',
                                         description='Combined data for all episodes')
                buf = io.BytesIO()
                pickle.dump(self.data_buffer, buf)
                buf.seek(0)
                with artifact.new_file("combined_data.pkl", mode="wb") as f:
                    f.write(buf.read())
                self.run.log_artifact(artifact)
                print(f"[W&B] Uploaded combined artifact with {len(self.data_buffer)} episodes")
                return True
            except Exception as e:
                attempts += 1
                wait_time = self.backoff_base ** attempts
                print(f"[W&B] Upload attempt {attempts} failed: {e}. Retrying in {wait_time:.1f}s")
                time.sleep(wait_time)
        print(f"[W&B] Failed to upload combined artifact after {self.max_retries} attempts")
        return False

    def finish(self):
        if self.run is not None:
            self.run.finish()


# Optional Weights & Biases
try:
    import wandb  # noqa
except Exception:
    wandb = None

# ------------------ Defaults / Globals ------------------ #
print("DEVICE", jax.devices())

DT = 0.1
DT_torch = 0.1
DELAY = 1
H = 8
i_start = 30
EP_LEN = 500

MPC = True
VIS = True
trajectory_type = "berlin_2018"
SIM = 'numerical'  # 'numerical' or 'unity'

if SIM == 'numerical':
    trajectory_type = "../../simulators/params-num.yaml"
    LF = 0.12
    LR = 0.24
    L = LF + LR

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='none', type=str, help='Name of the experiment')
parser.add_argument('--start_ep', type=int, default=1, help='Episode number to start from (inclusive)')
parser.add_argument('--end_ep', type=int, default=242, help='Episode number to end at (inclusive)')
parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging + artifact uploads')
parser.add_argument('--wandb_project', type=str, default='alpha-racer-data', help='W&B Project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='W&B Entity (team/user). Optional')
parser.add_argument('--checkpoint_dir', type=str, default='data', help='Directory to store per-episode files if not using W&B')
parser.add_argument('--mid_ckpt_steps', type=int, default=0, help='If >0, save mid-episode checkpoints every N steps (uploads when W&B enabled)')
parser.add_argument('--seed', type=int, default=-1, help='Global seed. If -1, uses current time')
parser.add_argument('--wandb_max_retries', type=int, default=3, help='Max retries for W&B uploads')
parser.add_argument('--wandb_retry_backoff', type=float, default=2.0, help='Exponential backoff base seconds for retries')
args = parser.parse_args()

# ------------------ Seeding Helpers ------------------ #
def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    # jax PRNG
    global key
    key = jax.random.PRNGKey(seed)
    print(f"[Seed] Global seed set to {seed}")

# Decide initial seed
_base_seed = args.seed if args.seed >= 0 else int(time.time())
seed_everything(_base_seed)

# Expand experiment name with controller type
exp_name = args.exp_name
if MPC:
    exp_name = exp_name + '_mpc'

# Ensure checkpoint directory exists (only used when not using W&B or for optional local backups)
os.makedirs(args.checkpoint_dir, exist_ok=True)

# Warn if W&B needed but not installed
if args.use_wandb and wandb is None:
    print("[W&B] wandb not installed. Disabling W&B.")
    args.use_wandb = False

# ------------------ Dynamics / Env setup ------------------ #
model_params_single = DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=DELAY)
model_params_single_opp = DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=DELAY)
model_params_single_opp1 = DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=-0., Ta=20., Tb=.0, mu=0.5, delay=DELAY)

dynamics_single = DynamicBicycleModel(model_params_single)
dynamics_single_opp = DynamicBicycleModel(model_params_single_opp)
dynamics_single_opp1 = DynamicBicycleModel(model_params_single_opp)

dynamics_single.reset()
dynamics_single_opp.reset()
dynamics_single_opp1.reset()

waypoint_generator = WaypointGenerator(trajectory_type, DT, H, 2.)
waypoint_generator_opp = WaypointGenerator(trajectory_type, DT, H, 1.)
waypoint_generator_opp1 = WaypointGenerator(trajectory_type, DT, H, 1.)

done = False

if SIM == 'numerical':
    env = OffroadCar({}, dynamics_single)
    env_opp = OffroadCar({}, dynamics_single_opp)
    env_opp1 = OffroadCar({}, dynamics_single_opp1)
    obs = env.reset(pose=[3., 5., -np.pi/2. - 0.72])
    obs_opp = env_opp.reset(pose=[0., 0., -np.pi/2. - 0.5])
    obs_opp1 = env_opp1.reset(pose=[-2., -6., -np.pi/2. - 0.3])

from mpc_controller import mpc
curr_steer = 0.

# ------------------ Helper: W&B upload with retries (from bytes) ------------------ #
def upload_artifact_from_buffer(exp_name_local, ep_no, buffer_bytes, description="", max_retries=3, backoff_base=2.0):
    """
    Uploads an in-memory bytes buffer as an artifact to W&B.
    buffer_bytes: io.BytesIO (position should be at 0).
    Returns True on success, False on failure.
    """
    if not args.use_wandb or wandb is None:
        return False
    attempts = 0
    while attempts < max_retries:
        try:
            # create artifact
            art_name = f"{exp_name_local}_ep{ep_no}"
            artifact = wandb.Artifact(name=art_name, type="dataset", description=description)
            # add the in-memory buffer as a file called episode.pkl inside artifact
            # Many wandb versions accept file_or_bytes kwarg

            buffer_bytes.seek(0)
            with artifact.new_file("episode.pkl", mode="wb") as f:
                f.write(buffer_bytes.read())
            wandb.log_artifact(artifact)
            return True
        except Exception as e:
            attempts += 1
            wait = backoff_base ** attempts
            print(f"[W&B] upload attempt {attempts} failed: {e}. Retrying after {wait:.1f}s ...")
            time.sleep(wait)
    print(f"[W&B] upload failed after {max_retries} attempts.")
    return False

# ------------------ Node ------------------ #
class CarNode(Node):
    def __init__(self):
        super().__init__('car_node')

        # --------------- Episode control --------------- #
        self.start_ep = args.start_ep
        self.end_ep = args.end_ep
        # Directly set episode number, no local resume-file logic needed
        self.ep_no = self.start_ep

        # Per-episode seed: base + ep_no
        self._set_episode_seed(self.ep_no)

        # --------------- W&B init (for single artifact) --------------- #
        self.use_wandb = args.use_wandb
        self.wandb_manager = None
        if self.use_wandb:
            self.wandb_manager = WandbArtifactManager(
                project=args.wandb_project,
                entity=args.wandb_entity,
                artifact_name=exp_name + "_combined",
                max_retries=args.wandb_max_retries,
                backoff_base=args.wandb_retry_backoff,
            )

        # --------------- ROS pubs --------------- #
        if VIS:
            self.path_pub_ = self.create_publisher(Path, 'path', 1)
            self.path_pub_nn = self.create_publisher(Path, 'path_nn', 1)
            self.path_pub_nn_opp = self.create_publisher(Path, 'path_nn_opp', 1)
            self.path_pub_nn_opp1 = self.create_publisher(Path, 'path_nn_opp1', 1)
            self.waypoint_list_pub_ = self.create_publisher(Path, 'waypoint_list', 1)
            self.left_boundary_pub_ = self.create_publisher(Path, 'left_boundary', 1)
            self.right_boundary_pub_ = self.create_publisher(Path, 'right_boundary', 1)
            self.raceline_pub_ = self.create_publisher(Path, 'raceline', 1)
            self.ref_trajectory_pub_ = self.create_publisher(Path, 'ref_trajectory', 1)
            self.pose_pub_ = self.create_publisher(PoseWithCovarianceStamped, 'pose', 1)
            self.odom_pub_ = self.create_publisher(Odometry, 'odom', 1)
            self.odom_opp_pub_ = self.create_publisher(Odometry, 'odom_opp', 1)
            self.odom_opp1_pub_ = self.create_publisher(Odometry, 'odom_opp1', 1)
            self.slow_timer_ = self.create_timer(10.0, self.slow_timer_callback)
            self.throttle_pub_ = self.create_publisher(Float64, 'throttle', 1)
            self.steer_pub_ = self.create_publisher(Float64, 'steer', 1)
            self.trajectory_array_pub_ = self.create_publisher(MarkerArray, 'trajectory_array', 1)
            self.body_pub_ = self.create_publisher(PolygonStamped, 'body', 1)
            self.body_opp_pub_ = self.create_publisher(PolygonStamped, 'body_opp', 1)
            self.body_opp1_pub_ = self.create_publisher(PolygonStamped, 'body_opp1', 1)
            self.status_pub_ = self.create_publisher(Int8, 'status', 1)

        # --------------- Controller state --------------- #
        self.raceline = waypoint_generator.raceline
        self.raceline_dev = waypoint_generator.raceline_dev

        self.last_i = -1
        self.last_i_opp = -1
        self.last_i_opp1 = -1
        self.L = LF + LR

        self.curr_speed_factor = 1.
        self.curr_lookahead_factor = 0.24
        self.curr_sf1 = 0.2
        self.curr_sf2 = 0.2
        self.blocking = 0.2

        self.curr_speed_factor_opp = 1.
        self.curr_lookahead_factor_opp = 0.15
        self.curr_sf1_opp = 0.1
        self.curr_sf2_opp = 0.5
        self.blocking_opp = 0.2

        self.curr_speed_factor_opp1 = 1.
        self.curr_lookahead_factor_opp1 = 0.15
        self.curr_sf1_opp1 = 0.1
        self.curr_sf2_opp1 = 0.5
        self.blocking_opp1 = 0.2

        self.states = []
        self.cmds = []
        self.i = 0
        self.curr_t_counter = 0.
        self.unity_state_new = [0., 0., 0., 0., 0., 0.]

        # per-episode buffer
        self.buffer = []

        # mid-episode ckpt tracking
        self.mid_ckpt_steps = max(0, args.mid_ckpt_steps)
        self._last_mid_ckpt = 0

        # episode stats
        self._ep_progress_ego = 0.0
        self._ep_progress_opp = 0.0
        self._ep_progress_opp1 = 0.0

    # ---------- Utility: find first missing file for resume (local) ---------- #
    def _resume_find_first_missing_episode(self, start_ep, end_ep):
        """
        Looks for local files and returns first missing episode number.
        NOTE: If you run with --use_wandb (no local files), set start_ep appropriately.
        """
        ep = start_ep
        while ep <= end_ep:
            fname = os.path.join(args.checkpoint_dir, f"{exp_name}_ep{ep}.pkl")
            if not os.path.exists(fname):
                return ep
            ep += 1
        return ep  # end + 1 means all exist

    # ---------- Utility: per-episode seed ---------- #
    def _set_episode_seed(self, ep_no: int):
        seed_everything(_base_seed + int(ep_no))
        print(f"[Episode {ep_no}] Using seed {_base_seed + int(ep_no)}")

    # ---------- Utility: save episode buffer (either local or direct W&B) ---------- #
    def _upload_current_episode(self):
        """
        Append the current episode's buffer to the single wandb artifact.
        """
        arr = np.array(self.buffer, dtype=np.float32)
        if self.use_wandb and self.wandb_manager is not None:
            self.wandb_manager.append_episode(arr)
            success = self.wandb_manager.upload_combined()
            if not success:
                print(f"[Episode {self.ep_no}] Warning: Failed to upload combined artifact")
            else:
                print(f"[Episode {self.ep_no}] Uploaded episode data to combined artifact")

    def _save_mid_episode(self):
        # Mid-episode checkpointing is disabled in this single-artifact workflow
        return

    def has_collided(self, px, py, theta, px_opp, py_opp, theta_opp, L=0.18, B=0.12):
        dx = px - px_opp
        dy = py - py_opp
        d_long = dx * np.cos(theta) + dy * np.sin(theta)
        d_lat = dy * np.cos(theta) - dx * np.sin(theta)
        cost1 = np.abs(d_long) - 2 * L
        cost2 = np.abs(d_lat) - 2 * B
        d_long_opp = dx * np.cos(theta_opp) + dy * np.sin(theta_opp)
        d_lat_opp = dy * np.cos(theta_opp) - dx * np.sin(theta_opp)
        cost3 = np.abs(d_long_opp) - 2 * L
        cost4 = np.abs(d_lat_opp) - 2 * B
        cost = (cost1 < 0) * (cost2 < 0) * (cost1 * cost2) + (cost3 < 0) * (cost4 < 0) * (cost3 * cost4)
        return cost

    def cbf_filter(self, s, s_opp, vs, vs_opp, sf1=0.3, sf2=0.3, lookahead_factor=1.0):
        eff_s = s_opp - s + (vs_opp - vs) * lookahead_factor
        factor = sf1 * np.exp(-sf2 * np.abs(eff_s))
        return factor

    def obs_state(self):
        return env.obs_state()

    def obs_state_opp(self):
        return env_opp.obs_state()

    def obs_state_opp1(self):
        return env_opp1.obs_state()

    def timer_callback(self):
        global obs, obs_opp, obs_opp1, curr_steer, s, s_opp, s_opp1
        try:
            ti = time.time()
            if SIM == 'unity' and (not getattr(self, 'pose_received', True)):
                return
            if SIM == 'unity' and (not getattr(self, 'vel_received', True)):
                return

            self.i += 1

            # ----- episode end handling ----- #
            if self.i > EP_LEN:
                print("Ego progress: ", self._ep_progress_ego)
                print("Opp1 progress: ", self._ep_progress_opp)
                print("Opp2 progress: ", self._ep_progress_opp1)

                # upload (or save) per-episode buffer (no local save when W&B enabled)
                self._upload_current_episode()

                # Prepare next episode
                self.last_i = -1
                self.last_i_opp = -1
                self.last_i_opp1 = -1
                self.ep_no += 1

                # Exit if finished
                if self.ep_no > self.end_ep:
                    print(f"[Done] Reached end_ep={self.end_ep}. Shutting down.")
                    if self.use_wandb and self.wandb_run:
                        self.wandb_run.finish()
                    rclpy.shutdown()
                    return

                # Re-seed for new episode
                self._set_episode_seed(self.ep_no)

                self.i = 1
                self.buffer = []
                self._last_mid_ckpt = 0

                waypoint_generator.last_i = -1
                waypoint_generator_opp.last_i = -1
                waypoint_generator_opp1.last_i = -1

                # Reset poses randomly in 3 configs
                choice = np.random.choice([0, 1, 2])
                if choice == 0:
                    obs_opp1 = env_opp1.reset(pose=[3., 5., -np.pi/2. - 0.72])
                    obs_opp = env_opp.reset(pose=[0., 0., -np.pi/2. - 0.5])
                    obs = env.reset(pose=[-2., -6., -np.pi/2. - 0.5])
                if choice == 1:
                    obs_opp1 = env_opp1.reset(pose=[3., 5., -np.pi/2. - 0.72])
                    obs = env.reset(pose=[0., 0., -np.pi/2. - 0.5])
                    obs_opp = env_opp.reset(pose=[-2., -6., -np.pi/2. - 0.5])
                if choice == 2:
                    obs = env.reset(pose=[3., 5., -np.pi/2. - 0.72])
                    obs_opp1 = env_opp1.reset(pose=[0., 0., -np.pi/2. - 0.5])
                    obs_opp = env_opp.reset(pose=[-2., -6., -np.pi/2. - 0.5])

                # Randomize params for next episode (same pattern as before)
                if self.ep_no % 3 == 0:
                    self.curr_sf1 = np.random.uniform(0.1, 0.5)
                    self.curr_sf2 = np.random.uniform(0.1, 0.5)
                    self.curr_lookahead_factor = np.random.uniform(0.12, 0.5)
                    self.curr_speed_factor = np.random.uniform(0.85, 1.1)
                    self.blocking = np.random.uniform(0., 1.0)
                elif self.ep_no % 3 == 1:
                    self.curr_sf1_opp = np.random.uniform(0.1, 0.5)
                    self.curr_sf2_opp = np.random.uniform(0.1, 0.5)
                    self.curr_lookahead_factor_opp = np.random.uniform(0.12, 0.5)
                    self.curr_speed_factor_opp = np.random.uniform(0.85, 1.1)
                    self.blocking_opp = np.random.uniform(0., 1.0)
                else:
                    self.curr_sf1_opp1 = np.random.uniform(0.1, 0.5)
                    self.curr_sf2_opp1 = np.random.uniform(0.1, 0.5)
                    self.curr_lookahead_factor_opp1 = np.random.uniform(0.12, 0.5)
                    self.curr_speed_factor_opp1 = np.random.uniform(0.85, 1.1)
                    self.blocking_opp1 = np.random.uniform(0., 1.0)

                print("ep_no:", self.ep_no)
                print("ego params: ", self.curr_sf1, self.curr_sf2, self.curr_lookahead_factor, self.curr_speed_factor)
                print("opp params: ", self.curr_sf1_opp, self.curr_sf2_opp, self.curr_lookahead_factor_opp, self.curr_speed_factor_opp)
                print("opp1 params:", self.curr_sf1_opp1, self.curr_sf2_opp1, self.curr_lookahead_factor_opp1, self.curr_speed_factor_opp1)

                # Reset episode progress counters
                self._ep_progress_ego = 0.0
                self._ep_progress_opp = 0.0
                self._ep_progress_opp1 = 0.0

            # ----- normal step ----- #
            mu_factor = 1.
            status = Int8()

            target_pos_tensor, _, s, e = waypoint_generator.generate(jnp.array(obs[:5]), dt=DT_torch, mu_factor=mu_factor)
            target_pos_tensor_opp, _, s_opp, e_opp = waypoint_generator_opp.generate(jnp.array(obs_opp[:5]), dt=DT_torch, mu_factor=mu_factor)
            target_pos_tensor_opp1, _, s_opp1, e_opp1 = waypoint_generator_opp1.generate(jnp.array(obs_opp1[:5]), dt=DT_torch, mu_factor=mu_factor)

            # update progress (for logging at episode end)
            self._ep_progress_ego = float(s)
            self._ep_progress_opp = float(s_opp)
            self._ep_progress_opp1 = float(s_opp1)

            curv = target_pos_tensor[0, 3]
            curv_opp = target_pos_tensor_opp[0, 3]
            curv_opp1 = target_pos_tensor_opp1[0, 3]

            curv_lookahead = target_pos_tensor[-1, 3]
            curv_opp_lookahead = target_pos_tensor_opp[-1, 3]
            curv_opp1_lookahead = target_pos_tensor_opp1[-1, 3]
            target_pos_list = np.array(target_pos_tensor)

            action = np.array([0., 0.])
            px, py, psi, vx, vy, omega = self.obs_state().tolist()
            theta = target_pos_list[0, 2]
            theta_diff = np.arctan2(np.sin(theta - psi), np.cos(theta - psi))
            px_opp, py_opp, psi_opp, vx_opp, vy_opp, omega_opp = self.obs_state_opp().tolist()
            px_opp1, py_opp1, psi_opp1, vx_opp1, vy_opp1, omega_opp1 = self.obs_state_opp1().tolist()
            theta_opp = target_pos_tensor_opp[0, 2]
            theta_diff_opp = np.arctan2(np.sin(theta_opp - psi_opp), np.cos(theta_opp - psi_opp))
            theta_opp1 = target_pos_tensor_opp1[0, 2]
            theta_diff_opp1 = np.arctan2(np.sin(theta_opp1 - psi_opp1), np.cos(theta_opp1 - psi_opp1))

            if self.i > i_start:
                if np.isnan(vx) or np.isnan(vy) or np.isnan(omega):
                    print("State received a nan value")
                    # save what we have so far (upload if W&B enabled)
                    self._save_and_upload_episode()
                    rclpy.shutdown()
                    return
                self.states.append([vx, vy, omega])
                if np.isnan(action[0]) or np.isnan(action[1]):
                    print("Action received a nan value")
                    self._save_and_upload_episode()
                    rclpy.shutdown()
                    return
                self.cmds.append([action[0], action[1]])

            if VIS:
                q = quaternion_from_euler(0, 0, psi)
                now = self.get_clock().now().to_msg()

                pose = PoseWithCovarianceStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = now
                pose.pose.pose.position.x = px
                pose.pose.pose.position.y = py
                pose.pose.pose.orientation.x = q[0]
                pose.pose.pose.orientation.y = q[1]
                pose.pose.pose.orientation.z = q[2]
                pose.pose.pose.orientation.w = q[3]
                self.pose_pub_.publish(pose)

                path = Path()
                path.header.frame_id = 'map'
                path.header.stamp = now
                for i_ in range(target_pos_list.shape[0]):
                    pose_st = PoseStamped()
                    pose_st.header.frame_id = 'map'
                    pose_st.pose.position.x = float(target_pos_list[i_][0])
                    pose_st.pose.position.y = float(target_pos_list[i_][1])
                    path.poses.append(pose_st)
                self.ref_trajectory_pub_.publish(path)

                mppi_path = Path()
                mppi_path.header.frame_id = 'map'
                mppi_path.header.stamp = now
                self.status_pub_.publish(status)

            if SIM == 'numerical':
                # ego
                if MPC:
                    steer, throttle, _, _, self.last_i = self.mpc(
                        (px, py, psi),
                        (s, e, vx),
                        (s_opp, e_opp, vx_opp),
                        (s_opp1, e_opp1, vx_opp1),
                        self.curr_sf1, self.curr_sf2,
                        self.curr_lookahead_factor * 2,
                        self.curr_speed_factor ** 2,
                        self.blocking, last_i=self.last_i)
                else:
                    steer, throttle, _, _, self.last_i = self.pure_pursuit(
                        (px, py, psi),
                        (s, e, vx),
                        (s_opp, e_opp, vx_opp),
                        (s_opp1, e_opp1, vx_opp1),
                        self.curr_sf1, self.curr_sf2,
                        self.curr_lookahead_factor,
                        self.curr_speed_factor,
                        self.blocking, last_i=self.last_i)

                if abs(e) > 0.55:
                    env.state.vx *= np.exp(-3 * (abs(e) - 0.55))
                    env.state.vy *= np.exp(-3 * (abs(e) - 0.55))
                    env.state.psi += (1 - np.exp(-(abs(e) - 0.55))) * (theta_diff)
                    steer += (-np.sign(e) - steer) * (1 - np.exp(-3 * (abs(e) - 0.55)))

                if abs(theta_diff) > 1.:
                    throttle += 0.2
                obs, reward, done, info = env.step(np.array([throttle, steer]))

                collision = self.has_collided(px, py, psi, px_opp, py_opp, psi_opp)
                collision1 = self.has_collided(px, py, psi, px_opp1, py_opp1, psi_opp1)
                collision2 = self.has_collided(px_opp, py_opp, psi_opp, px_opp1, py_opp1, psi_opp1)

                # opp
                if MPC:
                    steer, throttle, _, _, self.last_i_opp = self.mpc(
                        (px_opp, py_opp, psi_opp),
                        (s_opp, e_opp, vx_opp),
                        (s, e, vx),
                        (s_opp1, e_opp1, vx_opp1),
                        self.curr_sf1_opp, self.curr_sf2_opp,
                        self.curr_lookahead_factor_opp * 2,
                        self.curr_speed_factor_opp ** 2,
                        self.blocking_opp, last_i=self.last_i_opp)
                else:
                    steer, throttle, _, _, self.last_i_opp = self.pure_pursuit(
                        (px_opp, py_opp, psi_opp),
                        (s_opp, e_opp, vx_opp),
                        (s, e, vx),
                        (s_opp1, e_opp1, vx_opp1),
                        self.curr_sf1_opp, self.curr_sf2_opp,
                        self.curr_lookahead_factor_opp,
                        self.curr_speed_factor_opp,
                        self.blocking_opp, last_i=self.last_i_opp)

                if abs(e_opp) > 0.55:
                    env_opp.state.vx *= np.exp(-3 * (abs(e_opp) - 0.55))
                    env_opp.state.vy *= np.exp(-3 * (abs(e_opp) - 0.55))
                    env_opp.state.psi += (1 - np.exp(-(abs(e_opp) - 0.55))) * (theta_diff_opp)
                    steer += (-np.sign(e_opp) - steer) * (1 - np.exp(-3 * (abs(e_opp) - 0.55)))

                if abs(theta_diff_opp) > 1.:
                    throttle += 0.2
                action_opp = np.array([throttle, steer])
                obs_opp, reward, done, info = env_opp.step(action_opp)

                # opp1
                if MPC:
                    steer, throttle, _, _, self.last_i_opp1 = self.mpc(
                        (px_opp1, py_opp1, psi_opp1),
                        (s_opp1, e_opp1, vx_opp1),
                        (s, e, vx),
                        (s_opp, e_opp, vx_opp),
                        self.curr_sf1_opp1, self.curr_sf2_opp1,
                        self.curr_lookahead_factor_opp1 * 2,
                        self.curr_speed_factor_opp1 ** 2,
                        self.blocking_opp1, last_i=self.last_i_opp1)
                else:
                    steer, throttle, _, _, self.last_i_opp1 = self.pure_pursuit(
                        (px_opp1, py_opp1, psi_opp1),
                        (s_opp1, e_opp1, vx_opp1),
                        (s, e, vx),
                        (s_opp, e_opp, vx_opp),
                        self.curr_sf1_opp1, self.curr_sf2_opp1,
                        self.curr_lookahead_factor_opp1,
                        self.curr_speed_factor_opp1,
                        self.blocking_opp1, last_i=self.last_i_opp1)

                if abs(e_opp1) > 0.55:
                    env_opp1.state.vx *= np.exp(-3 * (abs(e_opp1) - 0.55))
                    env_opp1.state.vy *= np.exp(-3 * (abs(e_opp1) - 0.55))
                    env_opp1.state.psi += (1 - np.exp(-(abs(e_opp1) - 0.55))) * (theta_diff_opp1)
                    steer += (-np.sign(e_opp1) - steer) * (1 - np.exp(-3 * (abs(e_opp1) - 0.55)))
                if abs(theta_diff_opp1) > 1.:
                    throttle += 0.2
                action_opp1 = np.array([throttle, steer])
                obs_opp1, reward, done, info = env_opp1.step(action_opp1)

                state_obs = [
                    s, s_opp, s_opp1,
                    e, e_opp, e_opp1,
                    theta_diff, obs[3], obs[4], obs[5],
                    theta_diff_opp, obs_opp[3], obs_opp[4], obs_opp[5],
                    theta_diff_opp1, obs_opp1[3], obs_opp1[4], obs_opp1[5],
                    curv, curv_opp, curv_opp1,
                    curv_lookahead, curv_opp_lookahead, curv_opp1_lookahead,
                    self.curr_sf1, self.curr_sf2, self.curr_lookahead_factor, self.curr_speed_factor, self.blocking,
                    self.curr_sf1_opp, self.curr_sf2_opp, self.curr_lookahead_factor_opp, self.curr_speed_factor_opp, self.blocking_opp,
                    self.curr_sf1_opp1, self.curr_sf2_opp1, self.curr_lookahead_factor_opp1, self.curr_speed_factor_opp1, self.blocking_opp1
                ]

                diff_s = s_opp - s
                if diff_s < -75.: diff_s += 150.
                if diff_s > 75.: diff_s -= 150.

                diff_s1 = s_opp1 - s
                if diff_s1 < -75.: diff_s1 += 150.
                if diff_s1 > 75.: diff_s1 -= 150.

                diff_s2 = s_opp1 - s_opp
                if diff_s2 < -75.: diff_s2 += 150.
                if diff_s2 > 75.: diff_s2 -= 150.

                if diff_s > 0.:
                    env.state.vx *= np.exp(-20 * collision)
                    env.state.vy *= np.exp(-20 * collision)
                    env_opp.state.vx *= np.exp(-5 * collision)
                    env_opp.state.vy *= np.exp(-5 * collision)
                else:
                    env.state.vx *= np.exp(-5 * collision)
                    env.state.vy *= np.exp(-5 * collision)
                    env_opp.state.vx *= np.exp(-20 * collision)
                    env_opp.state.vy *= np.exp(-20 * collision)
                if collision > 0.:
                    print("Collision detected", s, s_opp, e, e_opp)

                if diff_s1 > 0.:
                    env.state.vx *= np.exp(-20 * collision1)
                    env.state.vy *= np.exp(-20 * collision1)
                    env_opp1.state.vx *= np.exp(-5 * collision1)
                    env_opp1.state.vy *= np.exp(-5 * collision1)
                else:
                    env.state.vx *= np.exp(-5 * collision1)
                    env.state.vy *= np.exp(-5 * collision1)
                    env_opp1.state.vx *= np.exp(-20 * collision1)
                    env_opp1.state.vy *= np.exp(-20 * collision1)
                if collision1 > 0.:
                    print("Collision detected", s, s_opp1, e, e_opp1)

                if diff_s2 > 0.:
                    env_opp.state.vx *= np.exp(-20 * collision2)
                    env_opp.state.vy *= np.exp(-20 * collision2)
                    env_opp1.state.vx *= np.exp(-5 * collision2)
                    env_opp1.state.vy *= np.exp(-5 * collision2)
                else:
                    env_opp.state.vx *= np.exp(-5 * collision2)
                    env_opp.state.vy *= np.exp(-5 * collision2)
                    env_opp1.state.vx *= np.exp(-20 * collision2)
                    env_opp1.state.vy *= np.exp(-20 * collision2)
                if collision2 > 0.:
                    print("Collision detected", s_opp, s_opp1, e_opp, e_opp1)

                self.buffer.append(state_obs)

            # mid-episode checkpoint if enabled
            self._save_mid_episode()

            w_pred_ = 0.
            _w_pred = 0.
            if VIS:
                now = self.get_clock().now().to_msg()
                q = quaternion_from_euler(0, 0, psi)

                odom = Odometry()
                odom.header.frame_id = 'map'
                odom.header.stamp = now
                odom.pose.pose.position.x = px
                odom.pose.pose.position.y = py
                odom.pose.pose.orientation.x = q[0]
                odom.pose.pose.orientation.y = q[1]
                odom.pose.pose.orientation.z = q[2]
                odom.pose.pose.orientation.w = q[3]
                odom.twist.twist.linear.x = vx
                odom.twist.twist.linear.y = vy
                odom.twist.twist.angular.z = omega
                odom.twist.twist.angular.x = w_pred_
                odom.twist.twist.angular.y = _w_pred
                self.odom_pub_.publish(odom)

                q_opp = quaternion_from_euler(0, 0, psi_opp)
                odom = Odometry()
                odom.header.frame_id = 'map'
                odom.header.stamp = now
                odom.pose.pose.position.x = px_opp
                odom.pose.pose.position.y = py_opp
                odom.pose.pose.orientation.x = q_opp[0]
                odom.pose.pose.orientation.y = q_opp[1]
                odom.pose.pose.orientation.z = q_opp[2]
                odom.pose.pose.orientation.w = q_opp[3]
                odom.twist.twist.linear.x = vx_opp
                odom.twist.twist.linear.y = vy_opp
                odom.twist.twist.angular.z = omega_opp
                self.odom_opp_pub_.publish(odom)

                throttle_msg = Float64()
                throttle_msg.data = float(action_opp[0])
                self.throttle_pub_.publish(throttle_msg)
                steer_msg = Float64()
                global curr_steer
                curr_steer += 1.0 * (float(action_opp[1]) - curr_steer)
                steer_msg.data = curr_steer
                self.steer_pub_.publish(steer_msg)

                # ego body
                pts = np.array([
                    [LF, L / 3],
                    [LF, -L / 3],
                    [-LR, -L / 3],
                    [-LR, L / 3],
                ])
                R = euler_matrix(0, 0, psi)[:2, :2]
                pts_w = (R @ pts.T).T + np.array([px, py])
                body = PolygonStamped()
                body.header.frame_id = 'map'
                body.header.stamp = now
                for i_ in range(pts_w.shape[0]):
                    p = Point32()
                    p.x = float(pts_w[i_, 0]); p.y = float(pts_w[i_, 1]); p.z = 0.
                    body.polygon.points.append(p)
                self.body_pub_.publish(body)

                # opp body
                pts = np.array([
                    [LF, L / 3],
                    [LF, -L / 3],
                    [-LR, -L / 3],
                    [-LR, L / 3],
                ])
                R = euler_matrix(0, 0, psi_opp)[:2, :2]
                pts_w = (R @ pts.T).T + np.array([px_opp, py_opp])
                body = PolygonStamped()
                body.header.frame_id = 'map'
                body.header.stamp = now
                for i_ in range(pts_w.shape[0]):
                    p = Point32()
                    p.x = float(pts_w[i_, 0]); p.y = float(pts_w[i_, 1]); p.z = 0.
                    body.polygon.points.append(p)
                self.body_opp_pub_.publish(body)

                # opp1 body
                pts = np.array([
                    [LF, L / 3],
                    [LF, -L / 3],
                    [-LR, -L / 3],
                    [-LR, L / 3],
                ])
                R = euler_matrix(0, 0, psi_opp1)[:2, :2]
                pts_w = (R @ pts.T).T + np.array([px_opp1, py_opp1])
                body = PolygonStamped()
                body.header.frame_id = 'map'
                body.header.stamp = now
                for i_ in range(pts_w.shape[0]):
                    p = Point32()
                    p.x = float(pts_w[i_, 0]); p.y = float(pts_w[i_, 1]); p.z = 0.
                    body.polygon.points.append(p)
                self.body_opp1_pub_.publish(body)

            tf = time.time()

        except Exception as e:
            print("Error in callback: ", e)
            # Attempt to save progress before exiting
            try:
                self._upload_current_episode()
            except Exception as ee:
                print("Failed to save on error:", ee)
            if self.use_wandb and self.wandb_run:
                try:
                    self.wandb_manager.finish()
                except Exception:
                    pass
            rclpy.shutdown()

    def get_curvature(self, x1, y1, x2, y2, x3, y3):
        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        s = (a + b + c) / 2
        return 4 * np.sqrt(max(0.0, s * (s - a) * (s - b) * (s - c))) / (a * b * c)

    def mpc(self, xyt, pose, pose_opp, pose_opp1, sf1, sf2, lookahead_factor, v_factor, blocking_factor, gap=0.06, last_i=-1):
        # (same logic as before) ...
        s, e, v = pose
        x, y, theta = xyt
        s_opp, e_opp, v_opp = pose_opp
        s_opp1, e_opp1, v_opp1 = pose_opp1

        # find closest on raceline
        if last_i == -1:
            dists = np.sqrt((self.raceline[:, 0] - x) ** 2 + (self.raceline[:, 1] - y) ** 2)
            closest_idx = np.argmin(dists)
        else:
            raceline_ext = np.concatenate((self.raceline[last_i:, :], self.raceline[:20, :]), axis=0)
            dists = np.sqrt((raceline_ext[:20, 0] - x) ** 2 + (raceline_ext[:20, 1] - y) ** 2)
            closest_idx = (np.argmin(dists) + last_i) % len(self.raceline)
        N_ = len(self.raceline)
        _e = self.raceline_dev[closest_idx]
        _e_opp = self.raceline_dev[(closest_idx + int((s_opp - s) / gap)) % N_]
        _e_opp1 = self.raceline_dev[(closest_idx + int((s_opp1 - s) / gap)) % N_]
        e = e + _e
        e_opp = e_opp + _e_opp
        e_opp1 = e_opp1 + _e_opp1

        curv = self.get_curvature(self.raceline[closest_idx - 1, 0], self.raceline[closest_idx - 1, 1],
                                  self.raceline[closest_idx, 0], self.raceline[closest_idx, 1],
                                  self.raceline[(closest_idx + 1) % len(self.raceline), 0],
                                  self.raceline[(closest_idx + 1) % len(self.raceline), 1])
        curr_idx = (closest_idx + 1) % len(self.raceline)
        next_idx = (curr_idx + 1) % len(self.raceline)
        next_dist = np.sqrt((self.raceline[next_idx, 0] - self.raceline[curr_idx, 0]) ** 2 +
                            (self.raceline[next_idx, 1] - self.raceline[curr_idx, 1]) ** 2)
        traj = []
        dist_target = 0
        for t in np.arange(0.1, 1.05, 0.1):
            dist_target += v_factor * self.raceline[curr_idx, 2] * 0.1

            shift2 = self.calc_shift(s, s_opp, v, v_opp, sf1, sf2, t)
            shift2 = np.abs(shift2) if e > e_opp else -np.abs(shift2)
            shift1 = self.calc_shift(s, s_opp1, v, v_opp1, sf1, sf2, t)
            shift1 = np.abs(shift1) if e > e_opp1 else -np.abs(shift1)
            shift = shift1 + shift2

            if abs(shift2) > abs(shift1):
                if (shift + e_opp) * shift < 0.:
                    shift = 0.
                else:
                    if abs(shift2) > 0.03:
                        shift += e_opp
            else:
                if (shift + e_opp1) * shift < 0.:
                    shift = 0.
                else:
                    if abs(shift1) > 0.03:
                        shift += e_opp1

            if abs(shift2) > abs(shift1):
                if (shift + e_opp) * shift < 0.:
                    shift = 0.
                else:
                    if abs(shift2) > 0.03:
                        shift += e_opp
            else:
                if (shift + e_opp1) * shift < 0.:
                    shift = 0.
                else:
                    if abs(shift1) > 0.03:
                        shift += e_opp1

            # closest agent logic
            dist_from_opp = s - s_opp
            if dist_from_opp < -75.: dist_from_opp += 150.
            if dist_from_opp > 75.: dist_from_opp -= 150.
            dist_from_opp1 = s - s_opp1
            if dist_from_opp1 < -75.: dist_from_opp1 += 150.
            if dist_from_opp1 > 75.: dist_from_opp1 -= 150.

            if dist_from_opp > 0 and (dist_from_opp < dist_from_opp1 or dist_from_opp1 < 0):
                bf = 1 - np.exp(-blocking_factor * max(v_opp - v, 0.))
                shift = shift + (e_opp - shift) * bf * self.calc_shift(s, s_opp, v, v_opp, sf1, sf2, t) / sf1
            elif dist_from_opp1 > 0 and (dist_from_opp1 < dist_from_opp or dist_from_opp < 0):
                bf = 1 - np.exp(-blocking_factor * max(v_opp1 - v, 0.))
                shift = shift + (e_opp1 - shift) * bf * self.calc_shift(s, s_opp1, v, v_opp1, sf1, sf2, t) / sf1

            while dist_target - next_dist > 0.:
                dist_target -= next_dist
                curr_idx = next_idx
                next_idx = (next_idx + 1) % len(self.raceline)
                next_dist = np.sqrt((self.raceline[next_idx, 0] - self.raceline[curr_idx, 0]) ** 2 +
                                    (self.raceline[next_idx, 1] - self.raceline[curr_idx, 1]) ** 2)
            ratio = dist_target / next_dist
            pt = (1. - ratio) * self.raceline[next_idx, :2] + ratio * self.raceline[curr_idx, :2]
            theta_traj = np.arctan2(self.raceline[next_idx, 1] - self.raceline[curr_idx, 1],
                                    self.raceline[next_idx, 0] - self.raceline[curr_idx, 0]) + np.pi / 2.
            shifted_pt = pt + shift * np.array([np.cos(theta_traj), np.sin(theta_traj)])
            traj.append(shifted_pt)

        lookahead_distance = lookahead_factor * self.raceline[curr_idx, 2]
        N = len(self.raceline)
        lookahead_idx = int(closest_idx + 5) % N
        lookahead_point = self.raceline[lookahead_idx]
        curv_lookahead = self.get_curvature(self.raceline[lookahead_idx - 1, 0], self.raceline[lookahead_idx - 1, 1],
                                            self.raceline[lookahead_idx, 0], self.raceline[lookahead_idx, 1],
                                            self.raceline[(lookahead_idx + 1) % N, 0],
                                            self.raceline[(lookahead_idx + 1) % N, 1])
        theta_traj = np.arctan2(self.raceline[(lookahead_idx + 1) % N, 1] - self.raceline[lookahead_idx, 1],
                                self.raceline[(lookahead_idx + 1) % N, 0] - self.raceline[lookahead_idx, 0]) + np.pi / 2.
        shifted_point = lookahead_point + shift * np.array([np.cos(theta_traj), np.sin(theta_traj), 0.])

        throttle, steer = mpc([x, y, theta, v], np.array(traj), lookahead_factor=lookahead_factor)

        alpha = theta - (theta_traj - np.pi / 2.)
        if alpha > np.pi:
            alpha -= 2 * np.pi
        if alpha < -np.pi:
            alpha += 2 * np.pi
        if np.abs(alpha) > np.pi / 6:
            steer = -np.sign(alpha)
        return steer, throttle, curv, curv_lookahead, closest_idx

    def calc_shift(self, s, s_opp, vs, vs_opp, sf1=0.4, sf2=0.1, t=1.0):
        if vs == vs_opp:
            return 0.
        ttc = (s_opp - s) + (vs_opp - vs) * t
        eff_s = ttc
        factor = sf1 * np.exp(-sf2 * np.abs(eff_s) ** 2)
        return factor

    def pure_pursuit(self, xyt, pose, pose_opp, pose_opp1, sf1, sf2, lookahead_factor, v_factor, blocking_factor, gap=0.06, last_i=-1):
        s, e, v = pose
        x, y, theta = xyt
        s_opp, e_opp, v_opp = pose_opp
        s_opp1, e_opp1, v_opp1 = pose_opp1
        if last_i == -1:
            dists = np.sqrt((self.raceline[:, 0] - x) ** 2 + (self.raceline[:, 1] - y) ** 2)
            closest_idx = np.argmin(dists)
        else:
            raceline_ext = np.concatenate((self.raceline[last_i:, :], self.raceline[:20, :]), axis=0)
            dists = np.sqrt((raceline_ext[:20, 0] - x) ** 2 + (raceline_ext[:20, 1] - y) ** 2)
            closest_idx = (np.argmin(dists) + last_i) % len(self.raceline)
        N_ = len(self.raceline)
        _e = self.raceline_dev[closest_idx]
        _e_opp = self.raceline_dev[(closest_idx + int((s_opp - s) / gap)) % N_]
        _e_opp1 = self.raceline_dev[(closest_idx + int((s_opp1 - s) / gap)) % N_]
        e = e + _e
        e_opp = e_opp + _e_opp
        e_opp1 = e_opp1 + _e_opp1
        shift2 = self.calc_shift(s, s_opp, v, v_opp, sf1, sf2, 0.5)
        shift2 = np.abs(shift2) if e > e_opp else -np.abs(shift2)
        shift1 = self.calc_shift(s, s_opp1, v, v_opp1, sf1, sf2, 0.5)
        shift1 = np.abs(shift1) if e > e_opp1 else -np.abs(shift1)
        shift = shift1 + shift2

        if abs(shift2) > abs(shift1):
            if (shift + e_opp) * shift < 0.:
                shift = 0.
            else:
                if abs(shift2) > 0.03:
                    shift += e_opp
        else:
            if (shift + e_opp1) * shift < 0.:
                shift = 0.
            else:
                if abs(shift1) > 0.03:
                    shift += e_opp1

        curv = self.get_curvature(self.raceline[closest_idx - 1, 0], self.raceline[closest_idx - 1, 1],
                                  self.raceline[closest_idx, 0], self.raceline[closest_idx, 1],
                                  self.raceline[(closest_idx + 1) % len(self.raceline), 0],
                                  self.raceline[(closest_idx + 1) % len(self.raceline), 1])
        lookahead_distance = lookahead_factor * self.raceline[closest_idx, 2]
        N = len(self.raceline)
        lookahead_idx = int(closest_idx + 1 + lookahead_distance // gap) % N
        e_ = -self.raceline_dev[lookahead_idx]
        if e_ + shift > 0.44:
            shift = 0.44 - e_
        if e_ + shift < -0.44:
            shift = -0.44 - e_
        lookahead_point = self.raceline[lookahead_idx]
        curv_lookahead = self.get_curvature(self.raceline[lookahead_idx - 1, 0], self.raceline[lookahead_idx - 1, 1],
                                            self.raceline[lookahead_idx, 0], self.raceline[lookahead_idx, 1],
                                            self.raceline[(lookahead_idx + 1) % N, 0],
                                            self.raceline[(lookahead_idx + 1) % N, 1])
        theta_traj = np.arctan2(self.raceline[(lookahead_idx + 1) % N, 1] - self.raceline[lookahead_idx, 1],
                                self.raceline[(lookahead_idx + 1) % N, 0] - self.raceline[lookahead_idx, 0]) + np.pi / 2.
        shifted_point = lookahead_point + shift * np.array([np.cos(theta_traj), np.sin(theta_traj), 0.])

        v_target = v_factor * lookahead_point[2]
        throttle = (v_target - v) + 9.81 * 0.1 * 4.65 / 20.
        _dx = shifted_point[0] - x
        _dy = shifted_point[1] - y

        dx = _dx * np.cos(theta) + _dy * np.sin(theta)
        dy = _dy * np.cos(theta) - _dx * np.sin(theta)
        alpha = np.arctan2(dy, dx)
        steer = 2 * self.L * dy / (dx ** 2 + dy ** 2)
        if np.abs(alpha) > np.pi / 2:
            steer = np.sign(dy)
        return steer, throttle, curv, curv_lookahead, closest_idx

    def slow_timer_callback(self):
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i_ in range(waypoint_generator.waypoint_list_np.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.waypoint_list_np[i_][0])
            pose.pose.position.y = float(waypoint_generator.waypoint_list_np[i_][1])
            path.poses.append(pose)
        self.waypoint_list_pub_.publish(path)

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i_ in range(waypoint_generator.left_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.left_boundary[i_][0])
            pose.pose.position.y = float(waypoint_generator.left_boundary[i_][1])
            path.poses.append(pose)
        self.left_boundary_pub_.publish(path)

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i_ in range(waypoint_generator.right_boundary.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.right_boundary[i_][0])
            pose.pose.position.y = float(waypoint_generator.right_boundary[i_][1])
            path.poses.append(pose)
        self.right_boundary_pub_.publish(path)

        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp = self.get_clock().now().to_msg()
        for i_ in range(waypoint_generator.raceline.shape[0]):
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = float(waypoint_generator.raceline[i_][0])
            pose.pose.position.y = float(waypoint_generator.raceline[i_][1])
            path.poses.append(pose)
        self.raceline_pub_.publish(path)


def main():
    rclpy.init()
    car_node = CarNode()
    # If resume concluded nothing to do, node may have shut down.
    if not rclpy.ok():
        return
    while rclpy.ok():
        car_node.timer_callback()
    car_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()








