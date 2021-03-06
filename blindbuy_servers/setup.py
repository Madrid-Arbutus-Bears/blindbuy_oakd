from setuptools import setup

package_name = 'blindbuy_servers'

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
            'product_distance_server = ' + package_name + '.product_distance_server:main',
            'interactive_product_server = ' + package_name + '.interactive_product_server:main',

        ],
    },
)
