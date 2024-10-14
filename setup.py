from setuptools import setup, find_packages

setup(
    name='tabs',
    description='Torsion Angular Bin Strings',
    version='1.0.0',
    py_modules=['tabs'],
    packages=['tabs'],
    install_requires=[],
    entry_points={},
    packages=find_packages(where="tabs"),
    package_dir={"": "tabs"},
    include_package_data=True,
    author="Jessica Braun",
    author_email="braunje@ethz.ch",
)
