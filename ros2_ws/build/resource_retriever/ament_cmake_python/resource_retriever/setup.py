from setuptools import find_packages
from setuptools import setup

setup(
    name='resource_retriever',
    version='3.1.3',
    packages=find_packages(
        include=('resource_retriever', 'resource_retriever.*')),
)
