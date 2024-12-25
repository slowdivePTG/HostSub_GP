from setuptools import setup, find_packages

setup(
    name="HostSub_GP",
    version="0.2",
    packages=find_packages(),  # Automatically find all packages and subpackages
    install_requires=[],  # Add dependencies here
    entry_points={
        "console_scripts": [
            "hostsub = hostsub_gp.scripts.hostsub:HostSub.entry_point"
        ]
    }
)