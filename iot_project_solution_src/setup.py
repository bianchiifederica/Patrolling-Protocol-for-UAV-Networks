import os
from glob import glob

from setuptools import setup
from setuptools import find_packages

package_name = 'iot_project_solution_src'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude='test'),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('lib', package_name), glob(package_name + '/math_utils.py*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_id',
    maintainer_email='your_email@studenti.uniroma1.it',
    description='Solution for the iot_project',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_controller=iot_project_solution_src.drone_controller:main',
            'task_assigner=iot_project_solution_src.task_assigner:main'
        ],
    },
)
