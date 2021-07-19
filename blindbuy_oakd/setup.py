import os
from glob import glob
from setuptools import setup
from setuptools import find_packages

package_name = 'blindbuy_oakd'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Include all launch files
        (os.path.join('share', package_name), glob('launch/*.py')),
        # Include rviz files
        (os.path.join('share', package_name), glob('rviz/*rviz')),
        # Include models files
        (os.path.join('share', package_name), glob('models/*blob')),
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
            'face_detection = ' + package_name + '.face_detection:main',
        ],
    },
)