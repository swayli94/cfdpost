from setuptools import setup, find_packages
from cfdpost import __version__, __name__

with open('Readme.md') as f:
      long_description = f.read()

setup(name=__name__,
      version=__version__,
      description='This is the module of CFD post procedures',
      long_description=long_description,
      keywords='CFD',
      download_url='https://github.com/swayli94/cfdpost/',
      license='MIT',
      author='Runze LI',
      author_email='swayli94@gmail.com',
      packages=find_packages(exclude=['example']),
      install_requires=['numpy', 'scipy'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ]
)

