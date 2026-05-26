# install the triple_pendulum package: pip install -e .

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
