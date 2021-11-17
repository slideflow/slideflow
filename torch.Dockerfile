# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN apt update

# Install necessary packages
RUN apt install -y liblapack-dev libblas-dev libgl1-mesa-glx libsm6 libxext6 wget vim g++ pkg-config libglib2.0-dev expat libexpat-dev libexif-dev libtiff-dev libgsf-1-dev openslide-tools libopenjp2-tools libpng-dev libtiff5-dev libjpeg-turbo8-dev libopenslide-dev
RUN sed -i '/^#\sdeb-src /s/^# *//' "/etc/apt/sources.list"
RUN apt update

# Build libvips 8.11 from source [slideflow requires 8.9+, latest deb in Ubuntu 18.04 is 8.4]
RUN apt install build-essential devscripts -y
RUN mkdir libvips
WORKDIR "/libvips"
RUN wget https://github.com/libvips/libvips/releases/download/v8.11.4/vips-8.11.4.tar.gz
RUN tar zxf vips-8.11.4.tar.gz
WORKDIR "/libvips/vips-8.11.4"
RUN ./configure
RUN make
RUN make install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

# Repair pixman
RUN mkdir scripts
WORKDIR "/scripts"
RUN wget https://github.com/jamesdolezal/slideflow/blob/master/pixman_repair.sh
RUN chmod +x pixman_repair.sh

# Install slideflow & download scripts
ENV SF_BACKEND=torch
RUN pip3 install slideflow pretrainedmodels
RUN wget https://raw.githubusercontent.com/jamesdolezal/slideflow/master/test.py
RUN wget https://raw.githubusercontent.com/jamesdolezal/slideflow/master/run_project.py
RUN wget https://raw.githubusercontent.com/jamesdolezal/slideflow/master/qupath_roi.groovy
RUN wget https://raw.githubusercontent.com/jamesdolezal/slideflow/master/qupath_roi_legacy.groovy
