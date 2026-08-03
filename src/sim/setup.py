from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        *[
            (str(Path('share') / package_name / Path(path).parent), [path])
            for path in glob('config/**/*.yaml', recursive=True)
        ],
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dstrahan',
    maintainer_email='dan.strahan08@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'chunk_data_collector = sim.chunk_data_collector:main',
            'stream_data_collector = sim.stream_data_collector:main',
            'sim_dataset_collector = sim.sim_dataset_collector:main',
            'fenceline_action_chunk_publisher = sim.fenceline_action_chunk_publisher:main',
            'fenceline_expert_trajectory = sim.fenceline_expert_trajectory:main',
            'road_expert_trajectory = sim.road_expert_trajectory:main',
            'shed_expert_trajectory = sim.shed_expert_trajectory:main',
            'generate_scene_task_specs = sim.generate_scene_task_specs:main',
            'validate_scene_task_specs = sim.validate_scene_task_specs:main',
            'expand_pose_variants = sim.expand_pose_variants:main',

        ],
    },
)
