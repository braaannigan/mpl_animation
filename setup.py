from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(name='mpl_animation',
      version='0.1',
      description='A module to create animations in matplotlib using an intuitive declarative syntax',
      url='https://github.com/braaannigan/mpl_animation',
      author='Liam Brannigan',
      author_email='braaannigan@gmail.com',
      license='MIT',
      packages=['mpl_animation'],
      install_requires=['numpy','matplotlib'],
      classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Visualization"
                    ],
      zip_safe=False)
