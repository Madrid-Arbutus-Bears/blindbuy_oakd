import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'blindbuy_ptz'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dani',
    maintainer_email='d.garcialopez@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_publisher = ' + package_name + '.joint_publisher:main',
            'ptz_publisher = ' + package_name + '.ptz_publisher:main',

        ],
    },
)
