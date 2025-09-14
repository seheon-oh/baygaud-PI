from pathlib import Path
from setuptools import setup, find_packages

# Long description from README.md (shown on PyPI)
root = Path(__file__).parent
readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""

# Runtime dependencies (copied from requirements.txt; build tools excluded)
install_requires = [
    "astropy==7.1.0",
    "astropy-iers-data==0.2025.9.8.0.36.17",
    "attrs==25.3.0",
    "casa-formats-io==0.3.0",
    "certifi==2025.8.3",
    "charset-normalizer==3.4.3",
    "click==8.2.1",
    "cloudpickle==3.1.1",
    "contourpy==1.3.3",
    "cycler==0.12.1",
    "dask==2025.9.0",
    "dynesty==2.1.5",
    "fitsio==1.2.6",
    "fonttools==4.59.2",
    "fsspec==2025.9.0",
    "idna==3.10",
    "imageio==2.37.0",
    "joblib==1.5.2",
    "jsonschema==4.25.1",
    "jsonschema-specifications==2025.9.1",
    "kiwisolver==1.4.9",
    "lazy_loader==0.4",
    "llvmlite==0.44.0",
    "locket==1.0.0",
    "matplotlib==3.10.6",
    "msgpack==1.1.1",
    "networkx==3.5",
    "numba==0.61.2",
    "numpy==2.2.6",
    "packaging==25.0",
    "partd==1.4.2",
    "pillow==11.3.0",
    "protobuf==6.32.0",
    "psutil==7.0.0",
    "pyerfa==2.0.1.5",
    "pyfiglet==1.0.4",
    "pyparsing==3.2.3",
    "python-dateutil==2.9.0.post0",
    "PyYAML==6.0.2",
    "radio-beam==0.3.9",
    "ray==2.49.1",
    "referencing==0.36.2",
    "requests==2.32.5",
    "rpds-py==0.27.1",
    "ruamel.yaml==0.18.15",
    "ruamel.yaml.clib==0.2.12",
    "scikit-image==0.25.2",
    "scipy==1.16.1",
    "six==1.17.0",
    "spectral-cube==0.6.6",
    "tifffile==2025.9.9",
    "tk==0.1.0",
    "toolz==1.0.0",
    "tqdm==4.67.1",
    "urllib3==2.5.0",
]


root = Path(__file__).parent
readme = (root / "README.md").read_text(encoding="utf-8")


# Distribution name can use hyphen; import package uses underscore.
setup(
    name="baygaud-pi",
    version="2.0.0",
    description="BAYesian GAUssian Decomposer for spectral-line profiles.",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Se-Heon Oh",
    author_email="seheon.oh@gmail.com",
    url="https://github.com/seheon-oh/baygaud-PI",
    license="MIT",
    python_requires=">=3.10",

    package_dir={"": "src"},
    packages=find_packages(where="src", include=["baygaud_pi", "baygaud_pi.*"]),
    include_package_data=True,

    install_requires=install_requires,

    entry_points={
        "console_scripts": [
            "baygaud-pi=baygaud_pi.baygaud:main",
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

