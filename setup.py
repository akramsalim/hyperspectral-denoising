from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read long description from README.md
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="hyperspectral-denoising",
    version="1.0.0",
    author="Akram Al-Saffah",
    author_email="akramsalim9@gmail.com",
    description="Self-Supervised Learning for Hyperspectral Image Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.tu-berlin.de/rsim/cv4rs-2024-winter/self-supervised-learning-for-hyperspectral-image-analysis/-/tree/akram?ref_type=heads",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'train-denoising=run_training:main',
        ],
    },
)