from setuptools import setup, find_packages

with open("README.md", "r") as rm:
    long_description = rm.read()

setup(
    name="troughfinder",
    version="0.0.2",
    author="Axel Guinot",
    author_email="axel.guinot.astro@gmail.com",
    description="Trough finder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aguinot/TroughFinder",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
