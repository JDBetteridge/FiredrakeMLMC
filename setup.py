from setuptools import setup, find_packages

setup(  name='mlmcparagen',
        version=1.0,
        author='Nick Twyman',
        author_email='nicholas.twyman15@imperial.ac.uk',
        description='',
        long_description='',
        url='',
        packages=find_packages(),
        install_requires = ['numpy', 'mpi4py'],
        classifiers=[   'Programming Language :: Python :: 3',
                        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                        'Operating System :: POSIX :: Linux'
                    ]
        )
