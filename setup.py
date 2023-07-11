from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

# Create a setup.py file to use this directory as a package. 

def get_requirements(source_path:str) -> List[str]:
    requirements=[]
    with open(source_path) as file_obj:
        requirements = file_obj.readlines()
        [req.replace('\n','') for req in requirements]
    
        if  HYPEN_E_DOT in requirements:
           requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name = 'Zomato Price Prediction',
    version = '0.1',
    author='Aqleem Khan',
    author_email='aqleemkhan408@gmail.com',
    install_requirements = get_requirements('requirements.txt'),
    packages = find_packages()
)