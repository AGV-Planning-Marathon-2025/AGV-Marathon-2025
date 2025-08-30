import numpy as np
from mpc_controller import mpc

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def has_collided(px, py, theta, px1, py1, theta1, L=0.18, B=0.12):
    dx = px - px1
    dy = py - py1
    d_long = dx * np.cos(theta) + dy * np.sin(theta)
    d_lat = dy * np.cos(theta) - dx * np.sin(theta)
    cost1 = np.abs(d_long) - 2 * L
    cost2 = np.abs(d_lat) - 2 * B
    d_long_opp = dx * np.cos(theta1) + dy * np.sin(theta1)
    d_lat_opp = dy * np.cos(theta1) - dx * np.sin(theta1)
    cost3 = np.abs(d_long_opp) - 2 * L
    cost4 = np.abs(d_lat_opp) - 2 * B
    cost = (cost1 < 0) * (cost2 < 0) * (cost1 * cost2) + (cost3 < 0) * (cost4 < 0) * (cost3 * cost4)
    return cost

def calc_shift(s,s_opp,vs,vs_opp,sf1=0.4,sf2=0.1,t=1.0) :
    # if s > s_opp :
    #     return 0.
    if vs == vs_opp :
        return 0.
    ttc = (s_opp-s)+(vs_opp-vs)*t
    eff_s = ttc 
    factor = sf1*np.exp(-sf2*np.abs(eff_s)**2)
    return factor

def get_curvature(x1, y1, x2, y2, x3, y3):
    a = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    b = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    c = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    s = (a+b+c)/2
    # prod is av vector cross product with bc vector
    prod = (x2-x1)*(y3-y2) - (x3-x2)*(y2-y1)
    return 4*np.sqrt(s*(s-a)*(s-b)*(s-c))/(a*b*c)*np.sign(prod)

def fix_difference(t):
    t = (t > 75.0435) * (t - 150.087) + (t < -75.0435) * (t + 150.087) + (t <= 75.0435) * (t >= -75.0435) * t
    return t

def mpc_caller(self, xyt, pose, pose_opp, pose_opp1, sf1, sf2, lookahead_factor, v_factor, blocking_factor, gap=0.06, last_i = -1) :
    s,e,v = pose
    x,y,theta = xyt
    s_opp,e_opp,v_opp = pose_opp
    s_opp1,e_opp1,v_opp1 = pose_opp1
    
    if last_i == -1:
        dists = np.sqrt((self.raceline[:,0]-x)**2 + (self.raceline[:,1]-y)**2)
        closest_idx = np.argmin(dists)
    else :
        raceline_ext = np.concatenate((self.raceline[last_i:,:],self.raceline[:20,:]),axis=0)
        dists = np.sqrt((raceline_ext[:20,0]-x)**2 + (raceline_ext[:20,1]-y)**2)
        closest_idx = (np.argmin(dists) + last_i)%len(self.raceline)
    N_ = len(self.raceline)
    _e = self.raceline_dev[closest_idx]
    _e_opp = self.raceline_dev[(closest_idx+int((s_opp-s)/gap))%N_]
    _e_opp1 = self.raceline_dev[(closest_idx+int((s_opp1-s)/gap))%N_]
    e = e + _e
    e_opp = e_opp + _e_opp
    e_opp1 = e_opp1 + _e_opp1
    curv = get_curvature(self.raceline[closest_idx-1,0],self.raceline[closest_idx-1,1],self.raceline[closest_idx,0],self.raceline[closest_idx,1],self.raceline[(closest_idx+1)%len(self.raceline),0],self.raceline[(closest_idx+1)%len(self.raceline),1])
    curr_idx = (closest_idx+1)%len(self.raceline)
    next_idx = (curr_idx+1)%len(self.raceline)
    next_dist = np.sqrt((self.raceline[next_idx,0]-self.raceline[curr_idx,0])**2 + (self.raceline[next_idx,1]-self.raceline[curr_idx,1])**2)
    traj = []
    dist_target = 0
    for t in np.arange(0.1,1.05,0.1) :
        dist_target += v_factor*self.raceline[curr_idx,2]*0.1
        
        shift2 = calc_shift(s,s_opp,v,v_opp,sf1,sf2,t)
        if e>e_opp :
            shift2 = np.abs(shift2)
        else :
            shift2 = -np.abs(shift2)
        shift1 = calc_shift(s,s_opp1,v,v_opp1,sf1,sf2,t)
        if e>e_opp1 :
            shift1 = np.abs(shift1)
        else :
            shift1 = -np.abs(shift1)
        shift = shift1 + shift2
        
        if abs(shift2) > abs(shift1) :
            if (shift+e_opp)*shift < 0. : 
                shift = 0.
            else :
                if abs(shift2) > 0.03:
                    shift += e_opp
        else :
            if (shift+e_opp1)*shift < 0. :
                shift = 0.
            else :
                if abs(shift1) > 0.03:
                    shift += e_opp1
        
        if abs(shift2) > abs(shift1) :
            if (shift+e_opp)*shift < 0. : 
                shift = 0.
            else :
                if abs(shift2) > 0.03:
                    shift += e_opp
        else :
            if (shift+e_opp1)*shift < 0. :
                shift = 0.
            else :
                if abs(shift1) > 0.03:
                    shift += e_opp1
    
        # Find the closest agent  
        dist_from_opp = s-s_opp
        if dist_from_opp < -75. : 
            dist_from_opp += 150. 
        if dist_from_opp > 75. :  
            dist_from_opp -= 150.
        dist_from_opp1 = s-s_opp1
        if dist_from_opp1 < -75. :
            dist_from_opp1 += 150.
        if dist_from_opp1 > 75. :
            dist_from_opp1 -= 150.
        if dist_from_opp>0 and (dist_from_opp < dist_from_opp1 or dist_from_opp1 < 0) :
            bf = 1 - np.exp(-blocking_factor*max(v_opp-v,0.))
            shift = shift + (e_opp-shift)*bf*calc_shift(s,s_opp,v,v_opp,sf1,sf2,t)/sf1
        elif dist_from_opp1>0 and (dist_from_opp1 < dist_from_opp or dist_from_opp < 0) :
            bf = 1 - np.exp(-blocking_factor*max(v_opp1-v,0.))
            shift = shift + (e_opp1-shift)*bf*calc_shift(s,s_opp1,v,v_opp1,sf1,sf2,t)/sf1
        
        while dist_target - next_dist > 0. :
            dist_target -= next_dist
            curr_idx = next_idx
            next_idx = (next_idx+1)%len(self.raceline)
            next_dist = np.sqrt((self.raceline[next_idx,0]-self.raceline[curr_idx,0])**2 + (self.raceline[next_idx,1]-self.raceline[curr_idx,1])**2)
        ratio = dist_target/next_dist
        pt = (1.-ratio)*self.raceline[next_idx,:2] + ratio*self.raceline[curr_idx,:2]
        theta_traj = np.arctan2(self.raceline[next_idx,1]-self.raceline[curr_idx,1],self.raceline[next_idx,0]-self.raceline[curr_idx,0]) + np.pi/2.
        shifted_pt = pt + shift*np.array([np.cos(theta_traj),np.sin(theta_traj)])
        traj.append(shifted_pt)
    
    
    lookahead_distance = lookahead_factor*self.raceline[curr_idx,2]
    N = len(self.raceline)
    lookahead_idx = int(closest_idx+5)%N
    lookahead_point = self.raceline[lookahead_idx]
    curv_lookahead = get_curvature(self.raceline[lookahead_idx-1,0],self.raceline[lookahead_idx-1,1],self.raceline[lookahead_idx,0],self.raceline[lookahead_idx,1],self.raceline[(lookahead_idx+1)%N,0],self.raceline[(lookahead_idx+1)%N,1])
    theta_traj = np.arctan2(self.raceline[(lookahead_idx+1)%N,1]-self.raceline[lookahead_idx,1],self.raceline[(lookahead_idx+1)%N,0]-self.raceline[lookahead_idx,0]) + np.pi/2.
    shifted_point = lookahead_point + shift*np.array([np.cos(theta_traj),np.sin(theta_traj),0.])
    
    throttle, steer = mpc([x,y,theta,v],np.array(traj),lookahead_factor=lookahead_factor)
    
    alpha = theta - (theta_traj - np.pi/2.)
    if alpha > np.pi :
        alpha -= 2*np.pi
    if alpha < -np.pi :
        alpha += 2*np.pi
    if np.abs(alpha) > np.pi/6 :
        steer = -np.sign(alpha) 
    return steer, throttle, curv, curv_lookahead, closest_idx

