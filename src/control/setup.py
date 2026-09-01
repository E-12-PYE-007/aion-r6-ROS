from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'basicmicro'],
    zip_safe=True,
    maintainer='dstrahan',
    maintainer_email='dan.strahan08@gmail.com',
    description='Motion control nodes for the Aion R6 rover -- turns target motion into motor commands, and the Roboclaw motor driver that consumes them.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'roboclaw_for_motors = control.roboclaw_for_motors:main',
            'pure_pursuit_controller = control.pure_pursuit_controller:main',
            'cmd_vel_to_roboclaw = control.cmd_vel_to_roboclaw:main',
        ],
    },
)
