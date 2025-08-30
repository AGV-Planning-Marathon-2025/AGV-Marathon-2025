import numpy as np
import jax.numpy as jnp

class CarManager:
    def __init__(self, idx, name, env, waypoint_gen, initial_pose, param_template):
        self.idx = idx
        self.name = name
        self.env = env
        self.waypoint_gen = waypoint_gen
        self.initial_pose = initial_pose
        self.reset(initial_pose)
        self.last_i = -1
        self.s, self.e = 0., 0.
        self.v = 0.
        
        self.params = dict(param_template)
        
        self.states = []
        self.cmds = []
        self.buffer = []
        self.regrets = []
        self.n_wins = 0

    def reset(self, pose):
        self.obs = self.env.reset(pose=pose)
        self.last_i = -1
        self.states.clear()
        self.cmds.clear()
        self.buffer.clear()
        self.regrets.clear()

    def get_obs(self):
        return self.env.obs_state().tolist()

    def get_waypoints(self, DT_torch, mu_factor, vx):
        return self.waypoint_gen.generate(jnp.array(self.obs[:5]), dt=DT_torch, mu_factor=mu_factor, body_speed=vx)

    def update_state(self):
        px, py, psi, vx, vy, omega = self.get_obs()
        self.s = self.e = self.v = None
        return px, py, psi, vx, vy, omega

    def log_step(self, px, py, psi, extra=None):
        entry = [px, py, psi]
        if extra is not None:
            entry += list(extra)
        self.buffer.append(entry)