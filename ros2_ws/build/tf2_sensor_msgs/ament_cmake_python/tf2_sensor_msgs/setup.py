from setuptools import find_packages
from setuptools import setup

setup(
    name='tf2_sensor_msgs',
    version='0.44.0',
    packages=find_packages(
        include=('tf2_sensor_msgs', 'tf2_sensor_msgs.*')),
)
