from setuptools import setup, find_packages

setup(
    name='visits-detector',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'keras',
        'haversine',
        'geoindex',
        'pandas',
        'h5py'
    ],
    author='Alexander Fridman',
    author_email='alexfridman@outlook.com',
)
