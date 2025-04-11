from setuptools import find_packages, setup



def parse_requirements(filename):
    with open(filename, "r") as file:
        return [
            line.strip() for line in file if line and not line.startswith("#")
        ]


requirements = parse_requirements("requirements.txt")

setup(
    name="BaldOrNot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    author="raodz, jakub1090cn, skrzypczykt",
    author_email="raodziem@gmail.com",
    description="A machine learning project for classifying individuals as bald or not based on facial images.",
    url="https://github.com/raodz/BaldOrNot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)
