# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages


requirements = [
    'numpy>1.18.0,<=1.20.0',
    'ray[default]>=1.0.0,<1.4.0',
    'scipy',
    'boto3'
]


test_requirements = [
    'pytest==6.1.1',
    'pytest-pylint==0.17.0',
    'moto==1.3.16',
    'coverage==5.3',
    'codecov==2.1.9',
    'tqdm'
]


__version__ = None


with open('nums/core/version.py') as f:
    # pylint: disable=exec-used
    exec(f.read(), globals())


with open("README.md", "r") as fh:
    long_description = fh.read()


def main():

    setup(
        name='nums',
        version=__version__,
        description="A numerical computing library for Python that scales.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/nums-project/nums",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: Unix",
        ],
        python_requires='>=3.6,<3.9',
        install_requires=requirements,
        extras_require={
            'testing': test_requirements
        },
        entry_points={
            'console_scripts': [
                'nums-coverage=nums.core.cmds.api_coverage:main',
            ],
        }
    )


if __name__ == "__main__":
    main()
