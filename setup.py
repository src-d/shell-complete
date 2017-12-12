import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 5, 0):
    typing = ["typing"]
else:
    typing = []

setup(
    name="shell-complete",
    description="Part of source{d}'s stack for machine learning on source "
                "code. Provides API and tools to train and use models for "
                "predicting next shell commands.",
    version="0.1.0",
    license="Apache 2.0",
    author="source{d}",
    author_email="machine-learning@sourced.tech",
    url="https://github.com/src-d/shell-complete",
    download_url='https://github.com/src-d/shell-complete',
    packages=find_packages(exclude=("shcomplete.tests",)),
    keywords=["machine learning on source code", "shell", "bash"],
    entry_points={
        "console_scripts": ["shcomplete=shcomplete.__main__:main"],
    },
    install_requires=["modelforge>=0.1.0-alpha", ] + typing,
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries"
    ]
)
