from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.3'

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='mlfromscratch',
    version=__version__,
    description='Python implementations of some of the fundamental Machine Learning models and algorithms from scratch.',
    url='https://github.com/eriklindernoren/ML-From-Scratch',
    download_url='https://github.com/eriklindernoren/ML-From-Scratch/tarball/master',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    author='Erik Linder-Noren',
    install_requires=install_requires,
    setup_requires=['numpy>=1.10', 'scipy>=0.17'],
    dependency_links=dependency_links,
    author_email='eriklindernoren@gmail.com'
)