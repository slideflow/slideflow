import versioneer
import setuptools
import pkg_resources


# Check for existing OpenCV installation
opencv_pkg = None
try:
    pkg_resources.get_distribution("opencv-python-headless")
    opencv_pkg = "opencv-python-headless"
except pkg_resources.DistributionNotFound:
    try:
        pkg_resources.get_distribution("opencv-python")
        opencv_pkg = "opencv-python"
    except pkg_resources.DistributionNotFound:
        opencv_pkg = "opencv-python-headless"  # Default to headless if neither is installed


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="slideflow",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="James Dolezal",
    author_email="james@slideflow.ai",
    description="Deep learning tools for digital histology",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/slideflow/slideflow",
    packages=setuptools.find_packages(),
    scripts=['scripts/slideflow-studio'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data = {
        'slideflow': [
            'norm/norm_tile.jpg',
            'slide/slideflow-logo-name-small.jpg',
            'studio/gui/fonts/DroidSans.ttf',
            'studio/gui/fonts/DroidSans-Bold.ttf',
            'studio/gui/icons/error.png',
            'studio/gui/icons/info.png',
            'studio/gui/icons/logo.png',
            'studio/gui/icons/success.png',
            'studio/gui/icons/warn.png',
            'studio/gui/icons/search.png',
            'studio/gui/icons/filter.png',
            'studio/gui/splash.png',
            'studio/gui/logo_dark_outline.png',
            'studio/gui/buttons/button_extensions_highlighted.png',
            'studio/gui/buttons/button_extensions_highlighted.png',
            'studio/gui/buttons/button_add_freehand.png',
            'studio/gui/buttons/button_add_polygon.png',
            'studio/gui/buttons/button_camera.png',
            'studio/gui/buttons/button_camera_highlighted.png',
            'studio/gui/buttons/button_circle_plus.png',
            'studio/gui/buttons/button_slide_highlighted.png',
            'studio/gui/buttons/button_project_highlighted.png',
            'studio/gui/buttons/small_button_cucim.png',
            'studio/gui/buttons/button_extensions_highlighted.png',
            'studio/gui/buttons/button_mosaic.png',
            'studio/gui/buttons/button_stylegan.png',
            'studio/gui/buttons/button_heatmap_highlighted.png',
            'studio/gui/buttons/button_cellseg.png',
            'studio/gui/buttons/button_floppy.png',
            'studio/gui/buttons/small_button_vips.png',
            'studio/gui/buttons/button_cellseg_highlighted.png',
            'studio/gui/buttons/small_button_folder.png',
            'studio/gui/buttons/button_mosaic_highlighted.png',
            'studio/gui/buttons/button_stylegan_highlighted.png',
            'studio/gui/buttons/small_button_ellipsis.png',
            'studio/gui/buttons/button_model_loaded_highlighted.png',
            'studio/gui/buttons/button_circle_lightning_highlighted.png',
            'studio/gui/buttons/small_button_lowmem.png',
            'studio/gui/buttons/button_circle_plus_highlighted.png',
            'studio/gui/buttons/button_heatmap.png',
            'studio/gui/buttons/button_extensions.png',
            'studio/gui/buttons/button_pencil_highlighted.png',
            'studio/gui/buttons/button_pencil.png',
            'studio/gui/buttons/small_button_ellipsis_highlighted.png',
            'studio/gui/buttons/button_circle_lightning.png',
            'studio/gui/buttons/button_folder_highlighted.png',
            'studio/gui/buttons/button_gear_highlighted.png',
            'studio/gui/buttons/button_model_loaded.png',
            'studio/gui/buttons/button_model_highlighted.png',
            'studio/gui/buttons/button_segment.png',
            'studio/gui/buttons/button_segment_highlighted.png',
            'studio/gui/buttons/small_button_verified.png',
            'studio/gui/buttons/button_model.png',
            'studio/gui/buttons/button_folder.png',
            'studio/gui/buttons/button_floppy_highlighted.png',
            'studio/gui/buttons/button_project.png',
            'studio/gui/buttons/small_button_refresh.png',
            'studio/gui/buttons/button_slide.png',
            'studio/gui/buttons/small_button_gear.png',
            'studio/gui/buttons/button_gear.png',
            'studio/gui/buttons/button_mil.png',
            'studio/gui/buttons/button_mil_highlighted.png',
            'gan/stylegan2/stylegan2/torch_utils/ops/bias_act.cpp',
            'gan/stylegan2/stylegan2/torch_utils/ops/bias_act.cu',
            'gan/stylegan2/stylegan2/torch_utils/ops/bias_act.h',
            'gan/stylegan2/stylegan2/torch_utils/ops/upfirdn2d.cpp',
            'gan/stylegan2/stylegan2/torch_utils/ops/upfirdn2d.cu',
            'gan/stylegan2/stylegan2/torch_utils/ops/upfirdn2d.h',
            'gan/stylegan3/stylegan3/torch_utils/ops/bias_act.cpp',
            'gan/stylegan3/stylegan3/torch_utils/ops/bias_act.cu',
            'gan/stylegan3/stylegan3/torch_utils/ops/bias_act.h',
            'gan/stylegan3/stylegan3/torch_utils/ops/upfirdn2d.cpp',
            'gan/stylegan3/stylegan3/torch_utils/ops/upfirdn2d.cu',
            'gan/stylegan3/stylegan3/torch_utils/ops/upfirdn2d.h',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu.cpp',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu.cu',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu.h',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu_ns.cu',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu_rd.cu',
            'gan/stylegan3/stylegan3/torch_utils/ops/filtered_lrelu_wr.cu',
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        'gast>=0.3.3',
        'scipy',
        'scikit-learn',
        'matplotlib>=3.2',
        'imageio',
        opencv_pkg,
        'shapely',
        'umap-learn',
        'seaborn<0.14',
        'pandas',
        'pyvips',
        'fpdf2',
        'scikit-image',
        'tqdm',
        'click',
        'protobuf<3.21',
        'tensorboard',
        'crc32c',
        'numpy',
        'tabulate',
		'rasterio',
        'smac==1.4.0',
        'ConfigSpace',
        'rich',
        'pillow>=6.0.0',
        'imgui>=2.0.0',
        'pyopengl',
        'glfw',
        'saliency',
        'pyperclip',
        'requests',
        'parameterized',
        'zarr',
        'gdown',
        'triangle',
        'pyarrow'
    ],
    extras_require={
        'tf': [
            'tensorflow>=2.7,<2.12',
            'tensorflow_probability<0.20',
            'tensorflow_datasets<4.9.0'
        ],
        'torch': [
            'torch',
            'torchvision',
            'pretrainedmodels',
            'cellpose<2.2',
            'spacy<3.8',
            'fastai',
            'pytorch-lightning',
            'timm',
            'segmentation-models-pytorch'
        ],
        'dev': [
            'sphinx',
            'sphinx-markdown-tables',
            'sphinxcontrib-video',
            'pygments-csv-lexer'
        ],
        'cucim': [
            'cucim'
        ],
        'cellpose': [
            'cellpose<2.2',
        ],
        'all': [
            'cellpose<2.2',
            'cucim',
            'sphinx',
            'sphinx-markdown-tables',
            'sphinxcontrib-video',
            'pygments-csv-lexer',
            'torch',
            'torchvision',
            'spacy<3.8',
            'fastai',
            'pretrainedmodels',
            'tensorflow>=2.7,<2.12',
            'tensorflow_probability<0.20',
            'tensorflow_datasets',
        ]
    },
)
