import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'ag_vla'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        # sys1_hw is the real executable: it pins the venv interpreter
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lisa',
    maintainer_email='lisa.yamamoto@student.unimelb.edu.au',
    description='AgVLA sys1 edge adapter: turns sys2 hidden states into action chunks.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cam_seq_bridge = ag_vla.cam_seq_bridge:main',
        ],
    },
)
