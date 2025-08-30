import rclpy
from rclpy.node import Node
import numpy as np
import torch
import time
import pickle
import argparse
import jax
import jax.numpy as jnp

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, Point32, PolygonStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64, Int8
from tf_transformations import quaternion_from_euler, euler_matrix

from car_dynamics.models_jax import DynamicBicycleModel, DynamicParams
from car_dynamics.envs.car3 import OffroadCar
from car_dynamics.controllers_jax import WaypointGenerator
from notebooks.dnn_model import DNNModel
from mpc_controller import mpc

from Inference.car_model import CarManager
from Inference.utils import has_collided, fix_difference
from Inference.parameter_update import grad_optim

DT = 0.1
DT_torch = 0.1
DELAY = 1
EP_LEN = 500
N_CAR = 3
i_start = 30
H = 9
LF, LR = 0.12, 0.24
L = LF + LR
car_names = ['ego', 'opp1', 'opp2']
reset_poses = [
    [3., 5., -np.pi/2.-0.72],
    [0., 0., -np.pi/2.-0.5],
    [-2., -6., -np.pi/2.-0.5]
]
param_template = dict(sf1=0.4, sf2=0.2, lookahead=0.35, speed=0.93, blocking=0.0)
trajectory_type = "../../simulators/params-num.yaml"

fs = 128

OPT_STYLE = 'grad'

files = '../q_models/value_model_'

V1 = DNNModel()
V2 = DNNModel()
V3 = DNNModel()
V1.load_state_dict(torch.load(f'{files}0.pth', map_location=torch.device('cpu')))
V2.load_state_dict(torch.load(f'{files}1.pth', map_location=torch.device('cpu')))
V3.load_state_dict(torch.load(f'{files}2.pth', map_location=torch.device('cpu')))

V_MODELS = [V1, V2, V3]
for V in V_MODELS:
    V.eval()

class CarNode(Node):
    def __init__(self):
        super().__init__('multi_car_mpc_node')
        self.cars: list[CarManager] = []
        for i, name in enumerate(car_names):
            dyn_model = DynamicBicycleModel(DynamicParams(num_envs=1, DT=DT, Sa=0.34, Sb=0., Ta=20., Tb=0., mu=0.5, delay=DELAY))
            env = OffroadCar({}, dyn_model)
            waypoint_gen = WaypointGenerator(trajectory_type, DT, H, 2. if i==0 else 1.)
            car = CarManager(i, name, env, waypoint_gen, reset_poses[i], param_template)
            self.cars.append(car)


        self.odom_pubs = [self.create_publisher(Odometry, f'odom_{name}', 1) for name in car_names]
        self.body_pubs = [self.create_publisher(PolygonStamped, f'body_{name}', 1) for name in car_names]
        self.status_pub = self.create_publisher(Int8, 'status', 1)
        self.ref_trajectory_pub = self.create_publisher(Path, 'ref_trajectory', 1)
        self.ep_no = 0
        self.n_wins = [0] * N_CAR
        self.i = 0
        self.dataset = []
        self.timer_ = self.create_timer(DT/3., self.timer_callback)
        self.exp_name = 'mpc_only'

    def reset_episode(self):
        order = [0,1,2]
        if self.ep_no < 34:
            pass
        elif self.ep_no < 67:
            order = [1,2,0]
        else:
            order = [2,0,1]
        for i, idx in enumerate(order):
            self.cars[i].reset(reset_poses[idx])
        self.i = 1

    def timer_callback(self):
        ti = time.time()
        self.i += 1

        # We reset the episode here
        if self.i > EP_LEN:
            s_values = [car.s for car in self.cars]
            max_idx = int(np.argmax(s_values))
            self.n_wins[max_idx] += 1
            
            for car in self.cars:
                car.last_i = -1
            
            self.ep_no += 1
            self.i = 1
            if self.ep_no < 34:
                for i, pose in enumerate(reset_poses):
                    self.cars[i].reset(pose)
            elif self.ep_no < 67:
                self.cars[1].reset(reset_poses[0])
                self.cars[2].reset(reset_poses[1])
                self.cars[0].reset(reset_poses[2])
            else:
                self.cars[2].reset(reset_poses[0])
                self.cars[0].reset(reset_poses[1])
                self.cars[1].reset(reset_poses[2])
            
            for car in self.cars:
                car.params['sf1'] = np.random.uniform(0.1,0.5)
                car.params['sf2'] = np.random.uniform(0.1,0.5)
                car.params['lookahead'] = np.random.uniform(0.12,0.5)
                car.params['speed'] = np.random.uniform(0.85,1.1)
                car.params['blocking'] = np.random.uniform(0.,1.0)
            
            self.dataset.append([car.buffer for car in self.cars])
            print("Saving dataset")
            regrets = {f'regrets{i+1}': car.regrets for i, car in enumerate(self.cars)}
            pickle.dump(regrets, open(f'regrets/regrets_{self.exp_name}.pkl','wb'))
            pickle.dump(self.dataset, open(f'recorded_races/racedata_{self.exp_name}.pkl','wb'))
            with open(f'n_wins/n_wins_{self.exp_name}.txt', 'w') as f:
                for item in self.n_wins:
                    f.write("%s\n" % item)
            if self.ep_no > 100:
                exit(0)
            return

        # This gatheers all waypoints and states
        pxs, pys, psis, vxs, vys, omegas = [], [], [], [], [], []
        s_vals, e_vals = [], []
        target_pos_tensors = []
        for idx, car in enumerate(self.cars):
            px, py, psi, vx, vy, omega = car.get_obs()
            pxs.append(px); pys.append(py); psis.append(psi); vxs.append(vx); vys.append(vy); omegas.append(omega)
            target_pos_tensor, _, s, e = car.get_waypoints(DT_torch, mu_factor=1., vx=vx)
            target_pos_tensors.append(target_pos_tensor)
            s_vals.append(s)
            e_vals.append(e)
            car.s = s
            car.e = e
            car.v = vx

        # Do MPC for all cars
        actions = []
        for idx, car in enumerate(self.cars):
            ego_idx = idx
            opp1_idx = (idx + 1) % N_CAR
            opp2_idx = (idx + 2) % N_CAR
            ego_pose = (s_vals[ego_idx], e_vals[ego_idx], vxs[ego_idx])
            opp1_pose = (s_vals[opp1_idx], e_vals[opp1_idx], vxs[opp1_idx])
            opp2_pose = (s_vals[opp2_idx], e_vals[opp2_idx], vxs[opp2_idx])
            xyt = (pxs[ego_idx], pys[ego_idx], psis[ego_idx])
            steer, throttle, _, _, car.last_i = car.env.node.mpc(xyt, ego_pose, opp1_pose, opp2_pose,
                                                                 car.params['sf1'], car.params['sf2'],
                                                                 car.params['lookahead'],
                                                                 car.params['speed'], car.params['blocking'],
                                                                 last_i=car.last_i)
            car.obs, reward, done, info = car.env.step(np.array([throttle, steer]))
            car.states.append([vxs[idx], vys[idx], omegas[idx]])
            car.cmds.append([throttle, steer])
            actions.append((steer, throttle))

        # This one checks for collissions
        collisions = np.zeros((N_CAR, N_CAR))
        for i in range(N_CAR):
            for j in range(i+1, N_CAR):
                collisions[i, j] = has_collided(pxs[i], pys[i], psis[i], pxs[j], pys[j], psis[j])
                collisions[j, i] = collisions[i, j]
        
        
        for i in range(N_CAR):
            for j in range(N_CAR):
                if i != j:
                    diff_s = s_vals[j] - s_vals[i]
                    if diff_s < -75.:
                        diff_s += 150.
                    if diff_s > 75.:
                        diff_s -= 150.
                    if diff_s > 0.:
                        self.cars[i].env.state.vx *= np.exp(-20*collisions[i,j])
                        self.cars[i].env.state.vy *= np.exp(-20*collisions[i,j])
                        self.cars[j].env.state.vx *= np.exp(-5*collisions[i,j])
                        self.cars[j].env.state.vy *= np.exp(-5*collisions[i,j])
                    else:
                        self.cars[i].env.state.vx *= np.exp(-5*collisions[i,j])
                        self.cars[i].env.state.vy *= np.exp(-5*collisions[i,j])
                        self.cars[j].env.state.vx *= np.exp(-20*collisions[i,j])
                        self.cars[j].env.state.vy *= np.exp(-20*collisions[i,j])
                    if collisions[i,j] > 0.:
                        print(f"Collision detected: {i},{j} s,e: {s_vals[i]},{s_vals[j]} {e_vals[i]},{e_vals[j]}")
        
        
        for idx, car in enumerate(self.cars):
            if self.i < 450:
                state_obs = []
                for i in range(N_CAR): state_obs.append(s_vals[i])
                for i in range(N_CAR): state_obs.append(e_vals[i])
                
                
                for i in range(N_CAR): 
                    theta = target_pos_tensors[i][0,2]
                    psi = psis[i]
                    theta_diff = np.arctan2(np.sin(theta-psi), np.cos(theta-psi))
                    state_obs.append(theta_diff)
                    state_obs += [vxs[i], vys[i], omegas[i]]
                
                for i in range(N_CAR):
                    curv = target_pos_tensors[i][0,3]
                    curv_lookahead = target_pos_tensors[i][-1,3]
                    state_obs += [curv, curv_lookahead]
                
                
                X = torch.tensor(np.array([state_obs])).float()
                for i in range(1, N_CAR):
                    X[:,i] -= float(X[:,0])
                    X[:,i] = fix_difference(X[:,i])


                if OPT_STYLE == 'grad':
                    param_indices = list(range(-15, 0))
                    bounds = {i: (0.1,0.5) for i in [-15,-14,-10,-9,-5,-4]}
                    bounds.update({i: (0.15,0.5) for i in [-13,-8,-3]})
                    bounds.update({i: (0.85,1.1) for i in [-12,-7,-2]})
                    bounds.update({i: (0.,1.) for i in [-11,-6,-1]})
                    X, _ = grad_optim(V_MODELS[idx], X, grad_rate=0.00001, param_indices=param_indices, bounds=bounds)

        for idx, car in enumerate(self.cars):
            car.log_step(pxs[idx], pys[idx], psis[idx], extra=[vxs[idx], vys[idx], omegas[idx], actions[idx][0], actions[idx][1]])
        
        for idx, car in enumerate(self.cars):
            self.publish_odom_and_body(idx, pxs[idx], pys[idx], psis[idx], vxs[idx])
        
        self.status_pub.publish(Int8(data=0))
        tf = time.time()
        print(f"Time taken: {tf-ti:.4f}s, episode {self.ep_no}, step {self.i}")

    def publish_odom_and_body(self, idx, px, py, psi, vx):
        q = quaternion_from_euler(0, 0, psi)
        odom = Odometry()
        odom.header.frame_id = 'map'
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]
        odom.twist.twist.linear.x = vx
        self.odom_pubs[idx].publish(odom)
        
        pts = np.array([
            [LF, L/3],
            [LF, -L/3],
            [-LR, -L/3],
            [-LR, L/3]
        ])
        
        R = euler_matrix(0, 0, psi)[:2, :2]
        pts = np.dot(R, pts.T).T + np.array([px, py])
        body = PolygonStamped()
        body.header.frame_id = 'map'
        body.header.stamp = odom.header.stamp
        for pt in pts:
            p = Point32(); p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.
            body.polygon.points.append(p)
        self.body_pubs[idx].publish(body)

def main():
    rclpy.init()
    node = CarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()