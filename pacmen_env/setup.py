from setuptools import setup, find_packages

setup(
    name='gym_foo',
    version='0.0.1',
    author="Chenghao Li",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=[
        "numpy",
        "gym>=0.15",
        "pyglet",
        "networkx",
    ],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
