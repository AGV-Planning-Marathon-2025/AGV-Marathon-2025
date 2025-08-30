import numpy as np

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


def fix_difference(t):
    t = (t > 75.0435) * (t - 150.087) + (t < -75.0435) * (t + 150.087) + (t <= 75.0435) * (t >= -75.0435) * t
    return t