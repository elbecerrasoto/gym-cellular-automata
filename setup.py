from setuptools import setup

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym_cellular_automata'))
from version import VERSION

setup(name='gym_cellular_automata',
      version=VERSION,
      description='Cellular Automata Environments for Reinforcement Learning following the OpenAI Gym API',
      url='https://github.com/elbecerrasoto/gym-cellular-automata',
      author='Emanuel Becerra Soto',
      author_email='elbecerrasoto@gmail.com',
      license='MIT',      
      install_requires=['gym', 'numpy', 'matplotlib', 'seaborn', 'yaml'],
      tests_require=['pytest'],
      python_requires='>=3.6',
)
