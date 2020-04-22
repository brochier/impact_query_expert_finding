from setuptools import setup, find_packages

import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
version = '0.0.0.0.0.0' # year.month.day.hour.minute.second
with open(os.path.join(current_folder,'VERSION')) as version_file:
    version = version_file.read().strip()

setup(name='impact_query_expert_finding',
      version=version,
      description='Python code of the paper "On the Impact of the Query Set on the Evaluation of Expert Finding Systems" presented in the workshop BIRNDL hosted at SIGIR 2018.',
      url='https://github.com/brochier/impact_query_expert_finding',
      author='Robin Brochier',
      author_email='robin.brochier@univ-lyon2.fr',
      license='MIT',
      include_package_data=True,
      packages=find_packages(exclude=['docs', 'tests*']),
      package_data={'': ['impact_query_expert_finding/resources/aminer/experts/*.txt', 'impact_query_expert_finding/conf.yml']},
      install_requires=[
          'numpy',
          'scipy',
          'pyyaml',
          'gensim',
          'scikit-learn',
          'matplotlib',
          'nltk',
          'patool'
      ],
      zip_safe=False)
