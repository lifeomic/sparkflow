from setuptools import setup
from setuptools import find_packages

setup(name='sparkflow',
      version='0.1',
      description='Deep learning on Spark with Tensorflow',
      keywords = ['tensorflow', 'spark', 'sparkflow', 'machine learning', 'lifeomic', 'deep learning'],
      url='',
      download_url='',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      install_requires=['tensorflow', 'flask', 'protobuf'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)