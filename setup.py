import slideflow
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="slideflow",
    version=slideflow.__version__,
    author="James Dolezal",
    author_email="james.dolezal@uchospitals.edu",
    description="Deep learning tools for digital histology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesdolezal/slideflow",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_data = {
        'slideflow': ['norm/norm_tile.jpg', 'slide/slideflow-logo-name-small.jpg'],
    },
    python_requires='>=3.7',
    install_requires=[
        'gast>=0.3.3',
        'scipy',
        'sklearn',
        'matplotlib>=3.2',
        'imageio',
        'opencv-python',
        'shapely',
        'umap-learn',
        'seaborn',
        'pandas',
        'pyvips',
        'fpdf',
        'spams',
        'lifelines',
        'scikit-image',
        'tqdm',
        'click',
        'protobuf',
        'tensorboard',
        'crc32c',
        'h5py',
        'numpy',
        'tabulate'
    ],
)
