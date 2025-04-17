from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()



setup(name='geofm-plant-traits',
      description='ML models for predicting plant traits from multispectral satellite data',
      url='https://github.com/isaachcamp/geofm-plant-traits',
      author='Isaac Campbell',
      author_email='isaac.campbell@wolfson.ox.ac.uk',
      license='GNU GPLv3',
      packages=find_packages(exclude=["tests", "docs"]),
      version="0.1.0",
      zip_safe=False
)
