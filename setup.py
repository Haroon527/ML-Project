from setuptools import setup, find_packages

setup(
    name="mlproject",
    version="0.0.1",
    author="Haroon",
    author_email="haroon.alrai@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
    ],
)
