from setuptools import setup
import movieTools

setup(name='movieTools',
      version=movieTools.__version__,
      description='Functions to handle time-lapse microscopy data',
      url='http://github.com/redst4r/movieTools/',
      author='https://raw.github.com/pytoolz/toolz/master/AUTHORS.md',
      maintainer='redst4r',
      maintainer_email='redst4r@web.de',
      license='BSD',
      keywords='timelapse microscopy',
      packages=['movieTools'],
      zip_safe=False)