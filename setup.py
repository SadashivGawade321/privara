from setuptools import setup, find_packages

setup(
    name="privara",
    version="1.0.0",
    description="Privara: Privacy-Preserving Federated Learning Simulator",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
