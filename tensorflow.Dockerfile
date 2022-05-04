# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:2.8.0-gpu

# Install necessary packages
RUN apt update && \
    apt install -y liblapack-dev libblas-dev libgl1-mesa-glx libsm6 libxext6 wget vim g++ pkg-config libglib2.0-dev expat libexpat-dev libexif-dev libtiff-dev libgsf-1-dev openslide-tools libopenjp2-tools libpng-dev libtiff5-dev libjpeg-turbo8-dev libopenslide-dev && \
    sed -i '/^#\sdeb-src /s/^# *//' "/etc/apt/sources.list" && \
    apt update

# Build libvips 8.11 from source [slideflow requires 8.9+, latest deb in Ubuntu 18.04 is 8.4]
RUN apt install build-essential devscripts -y && \
    mkdir libvips && \
    mkdir scripts
WORKDIR "/libvips"
RUN wget https://github.com/libvips/libvips/releases/download/v8.11.4/vips-8.11.4.tar.gz && \
    tar zxf vips-8.11.4.tar.gz
WORKDIR "/libvips/vips-8.11.4"
RUN ./configure && make && make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Repair pixman
WORKDIR "/scripts"
RUN wget https://github.com/jamesdolezal/slideflow/blob/1.1.3/pixman_repair.sh && \
    chmod +x pixman_repair.sh

# Install slideflow & download scripts
RUN pip3 install slideflow==1.1.3 && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/1.1.3/test.py && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/1.1.3/run_project.py && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/1.1.3/qupath_roi.groovy && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/1.1.3/qupath_roi_legacy.groovy && \
    pip3 install spams --no-cache --force-reinstall --no-deps
