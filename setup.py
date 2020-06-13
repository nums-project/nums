import sys
from setuptools import setup, find_packages
import nums

requirements = [
    'numpy',
    'boto3',
    'joblib',
    'tqdm',
    'pytest',
    'xxhash',
    'networkx'
]


with open("README.md", "r") as fh:
    long_description = fh.read()


def main():

    if sys.version_info[:2] < (3, 6):
        raise RuntimeError("Python version >= 3.6 required.")

    import numc

    setup(
        name='nums',
        version=nums.__version__,
        author='Melih Elibol',
        author_email="elibol@gmail.com",
        description="A numerical computing library for python that scales.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/nums-project/nums",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Unix",
        ],
        python_requires='>=3.6',
        install_requires=requirements,
    )


if __name__ == "__main__":
    main()
