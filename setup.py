from setuptools import setup, find_packages, command

from setuptools.command.develop import develop
from setuptools.command.install import install


def post_install():
    from madpack.doc import run_notebook
    print('generate documentation...')
    try:
        run_notebook('doctests/documentation.ipynb')
        print('done. Check doctests/documentation.html')
    except BaseException as e:
        print('Creating the documentation failed. Probably some soft dependency is missing '
              '(e.g. nbformat, nbconvert and pygments). This is no problem, madpack will'
              'still work.')


class PostDevelop(develop):
    def run(self):
        develop.run(self)
        post_install()


class PostInstall(install):
    def run(self):
        install.run(self)
        post_install()


setup(
    name='madpack',
    version='0.1.0',
    cmdclass={
        'develop': PostDevelop,
        'install': PostInstall,
    },
    packages=find_packages(),
    install_requires=[
        'numpy>=1.17',
        'scipy>=1.2.1',
        'matplotlib>=3.0.3',
        'torch>=1.1',
        'torchvision>=0.3',
        'scikit-image>=0.14.2',
        'fire',
        'pyyaml>=5.1',
        'pillow>=7',
        'markdown',
    ],
)