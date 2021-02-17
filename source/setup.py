import slideflow
import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slideflow",
    version=slideflow.__version__,
    author="James Dolezal",
    author_email="james.dolezal@uchospitals.edu",
    description="Tools for deep learning from tumor histology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pearson-laboratory/slideflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
	package_data = {
		'slideflow': ['util/norm_tile.jpg']
	},
    python_requires='>=3.7',
)