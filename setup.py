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
        'slideflow': [
            'norm/norm_tile.jpg',
            'slide/slideflow-logo-name-small.jpg',
            'gan/stylegan2/torch_utils/ops/bias_act.cpp',
            'gan/stylegan2/torch_utils/ops/bias_act.cu',
            'gan/stylegan2/torch_utils/ops/bias_act.h',
            'gan/stylegan2/torch_utils/ops/upfirdn2d.cpp',
            'gan/stylegan2/torch_utils/ops/upfirdn2d.cu',
            'gan/stylegan2/torch_utils/ops/upfirdn2d.h',
            'gan/stylegan3/torch_utils/ops/bias_act.cpp',
            'gan/stylegan3/torch_utils/ops/bias_act.cu',
            'gan/stylegan3/torch_utils/ops/bias_act.h',
            'gan/stylegan3/torch_utils/ops/upfirdn2d.cpp',
            'gan/stylegan3/torch_utils/ops/upfirdn2d.cu',
            'gan/stylegan3/torch_utils/ops/upfirdn2d.h',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu.cpp',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu.cu',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu.h',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu_ns.cu',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu_rd.cu',
            'gan/stylegan3/torch_utils/ops/filtered_lrelu_wr.cu',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'gast>=0.3.3',
        'scipy',
        'scikit-learn>=1.1',
        'matplotlib>=3.2',
        'imageio',
        'opencv-python',
        'shapely',
        'umap-learn',
        'seaborn<0.12',
        'pandas',
        'pyvips',
        'fpdf',
        'lifelines',
        'scikit-image',
        'tqdm',
        'click',
        'protobuf<=3.20.2',
        'tensorboard',
        'crc32c',
        'h5py',
        'numpy<1.22',
        'tabulate',
		'rasterio',
        'smac',
        'ConfigSpace',
        'pyarrow',
        'ninja',
        'rich',
        'pillow>=6.0.0',
        'imgui',
        'pyopengl',
        'glfw',
        'saliency',
        'pyperclip'
    ],
    extras_require={
        'tf': [
            'tensorflow>=2.7,<2.10',
            'tensorflow_probability'
        ],
        'torch': [
            'torch',
            'torchvision',
            'pretrained_models'
        ],
        'dev': [
            'parameterized',
            'sphinx',
            'sphinx-markdown-tables',
            'sphinxcontrib-video'
        ]
    },
)
