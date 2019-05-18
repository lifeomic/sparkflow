from setuptools import setup
from setuptools import find_packages

setup(name='sparkflow',
      version='0.7.0',
      description='Deep learning on Spark with Tensorflow',
      keywords = ['tensorflow', 'spark', 'sparkflow', 'machine learning', 'lifeomic', 'deep learning'],
      url='https://github.com/lifeomic/sparkflow',
      download_url='https://github.com/lifeomic/sparkflow/archive/0.7.0.tar.gz',
      author='Derek Miller',
      author_email='dmmiller612@gmail.com',
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      install_requires=['tensorflow', 'flask', 'protobuf', 'requests', 'dill'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
