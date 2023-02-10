from setuptools import setup, find_packages

setup(name='rnn_coach',
      version='0.1',
      description='training RNNs with various activation functions on user-defined tasks. Anaylzing them with fixed and slow points analysis.',
      url='https://github.com/engellab/rnn-coach',
      author='Pavel Tolmachev',
      author_email='pt1290@princeton.edu',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)