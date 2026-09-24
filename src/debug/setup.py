import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'debug'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'matplotlib'],
    zip_safe=True,
    maintainer='dstrahan',
    maintainer_email='dan.strahan08@gmail.com',
    description='Debug and simulation nodes for testing the aion-r6 action-chunk pipeline',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simulate_action_chunk = debug.simulate_action_chunk:main',
            'roboclaw_tests = debug.roboclaw_tests:main',
            'path_plotter = debug.path_plotter:main',
            'sim_robot = debug.sim_robot:main',
            'static_esdf_publisher = debug.static_esdf_publisher:main',
            'chunk_generator = debug.chunk_generator:main',
        ],
    },
)
