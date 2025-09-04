import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yug/agv-alpharacer/alpha-RACER/ros2_ws/install/examples_tf2_py'
