import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'weed_geotagging'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # weed_detector_hw pins the ag_sys1 venv interpreter (CUDA torch); see the script.
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dstrahan',
    maintainer_email='dan.strahan08@gmail.com',
    description='Project weed detections through depth into robot/map coordinates.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'weed_geotagger = weed_geotagging.weed_geotagger:main',
            'yolo_weed_detector = weed_geotagging.yolo_weed_detector:main',
            'rover_visualiser = weed_geotagging.rover_visualiser:main',
        ],
    },
)
