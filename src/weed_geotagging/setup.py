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
            'fake_weed_inputs = weed_geotagging.fake_weed_inputs:main',
            'yolo_weed_detector = weed_geotagging.yolo_weed_detector:main',
        ],
    },
)
