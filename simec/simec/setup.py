from setuptools import setup 
  
setup( 
    name='simec', 
    version='0.1', 
    description='Python package for SIMEC and SIMEXP algorithms', 
    author='Elisabetta Rocchetti', 
    author_email='elisabetta.rocchetti@unimi.it', 
    packages=['simec'], 
    install_requires=[ 
        'torch' 
    ], 
) 