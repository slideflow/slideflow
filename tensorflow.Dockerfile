# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.11.1-gpu
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install necessary packages
RUN apt update && \
    apt install -y liblapack-dev libblas-dev libgl1-mesa-glx libsm6 libxext6 wget vim g++ pkg-config libglib2.0-dev expat libexpat-dev libexif-dev libtiff-dev libgsf-1-dev openslide-tools libopenjp2-tools libpng-dev libtiff5-dev libjpeg-turbo8-dev libopenslide-dev python3-tk python3-dev && \
    sed -i '/^#\sdeb-src /s/^# *//' "/etc/apt/sources.list" && \
    apt update && \
    mkdir /scripts

# Build libvips 8.12 from source [slideflow requires 8.9+, latest deb in Ubuntu 18.04 is 8.4]
RUN apt install build-essential devscripts -y && \
    mkdir /libvips && \
    cd /libvips && \
    wget https://github.com/libvips/libvips/releases/download/v8.12.2/vips-8.12.2.tar.gz && \
    tar zxf vips-8.12.2.tar.gz && \
    cd /libvips/vips-8.12.2 && \
    ./configure && make && make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV SF_VIPS_VERSION=8.12.2

# Repair pixman
WORKDIR "/scripts"
RUN wget https://raw.githubusercontent.com/slideflow/slideflow/2.0.1/scripts/pixman_repair.sh && \
    chmod +x pixman_repair.sh && \
    ./pixman_repair.sh

# Install slideflow & download scripts
ENV SF_BACKEND=tensorflow
RUN pip3 install --upgrade setuptools pip wheel
RUN pip3 install slideflow[cucim]==3.0.2 cupy-cuda11x tensorflow_datasets tensorflow_probability==0.17.* versioneer && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/test.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/slideflow-studio.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/run_project.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/qupath_roi.groovy && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/qupath_roi_legacy.groovy && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/master/scripts/welcome.py && \
    chmod +x /scripts/welcome.py

# Set entrypoint
ENTRYPOINT ["bash", "-c", "python3 /scripts/welcome.py && exec /bin/bash"]
