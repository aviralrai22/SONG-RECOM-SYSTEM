from setuptools import find_packages,setup




#project basic information so that if someone want to install the project as ans package can read the information
 
setup(
name='machine learning project on songs_recommendation_system',
version= '0.0.1',
author='aviral rai',
author_email='aviralrai22@gmail.com',
#this will automatically finds the folders need to install
packages=find_packages(),
install_requires=['numpy','pandas']

)