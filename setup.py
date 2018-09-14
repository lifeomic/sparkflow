from setuptools import setup
from setuptools import find_packages

setup(name='sparkflow',
      version='0.3.1',
      description='Deep learning on Spark with Tensorflow',
      keywords = ['tensorflow', 'spark', 'sparkflow', 'machine learning', 'lifeomic', 'deep learning'],
      url='https://github.com/lifeomic/sparkflow',
      download_url='https://github.com/lifeomic/sparkflow/archive/0.3.1.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      install_requires=['tensorflow', 'flask', 'protobuf', 'requests'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)