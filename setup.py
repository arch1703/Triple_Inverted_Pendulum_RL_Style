"""
setup.py – Install the triple_pendulum package into the active conda env.

Run once from the project root:
    pip install -e .

This makes ``import triple_pendulum`` work from any script without
manually manipulating sys.path.
"""

from setuptools import find_packages, setup

setup(
    name="triple-pendulum",
    version="1.0.0",
    description="Triple inverted pendulum RL environment for Isaac Lab",
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    python_requires=">=3.10",
    install_requires=[
        "skrl[torch]>=2.0.0",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "imageio[ffmpeg]",
        "numpy",
        "scipy",
    ],
)
