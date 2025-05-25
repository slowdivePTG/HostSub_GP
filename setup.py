from setuptools import setup, find_packages

setup(
    name="HostSub_GP",
    version="0.2",
    packages=find_packages(),  # Automatically find all packages and subpackages
    package_data={"hostsub_gp": ["data/*"]},
    include_package_data=True,
    install_requires=["tinygp", "jaxopt", "astroquery", "reproject"],  # Core dependencies
    extras_require={
        "pypeit": ["pypeit"],  # Optional dependency for PypeIt integration
        "all": ["pypeit"],  # Install all optional dependencies
    },
    entry_points={
        "console_scripts": [
            "hostsub = hostsub_gp.scripts.hostsub:HostSub.entry_point"
        ]
    }
)
