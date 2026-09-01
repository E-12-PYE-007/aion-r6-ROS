from setuptools import find_packages, setup

package_name = 'localisation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dstrahan',
    maintainer_email='dan.strahan08@gmail.com',
    description='State-estimation nodes for the Aion R6 rover -- turns raw sensor sources into pose/velocity estimates.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'encoder_localisation = localisation.encoder_localisation:main',
            'cam_slam_bridge = localisation.cam_slam_bridge:main',
        ],
    },
)
