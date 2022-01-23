



import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()


thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()


setup(
    name="budget_optimisation",
    version="0.0.1",
    author="Deza Pasquale",
    author_email="camila.deza.pasquale@gmail.com, facujdeza@gmail.com",
    description="Channel optimisation",
    long_description="Optimisation of your marketing investment to maximize your KPI ",
    long_description_content_type="text/markdown",
    url="https://github.com/facundodeza/budget_optimisation",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
   
)
