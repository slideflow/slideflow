# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
RUN apt update
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install necessary packages
RUN apt update && \
    apt install -y liblapack-dev libblas-dev libgl1-mesa-glx libsm6 libxext6 wget vim g++ pkg-config libglib2.0-dev expat libexpat-dev libexif-dev libtiff-dev libgsf-1-dev openslide-tools libopenjp2-tools libpng-dev libtiff5-dev libjpeg-turbo8-dev libopenslide-dev python3-tk && \
    sed -i '/^#\sdeb-src /s/^# *//' "/etc/apt/sources.list" && \
    apt update && \
    mkdir /scripts

# Build libvips 8.15 from source [slideflow requires 8.9+]
RUN apt install build-essential devscripts meson -y && \
    mkdir /libvips && \
    cd /libvips && \
    wget https://github.com/libvips/libvips/archive/refs/tags/v8.15.3.tar.gz && \
    tar zxf v8.15.3.tar.gz && \
    cd /libvips/libvips-8.15.3 && \
    meson setup build --prefix /usr/local/lib && \
    cd /libvips/libvips-8.15.3/build && \
    meson compile && \
    meson install
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV SF_VIPS_VERSION=8.15.3

# Apply pytorch CuDNN patch
WORKDIR "/opt/conda/lib"
RUN ln -sn libnvrtc.so.11.8.89 libnvrtc.so

# Install slideflow & download scripts
WORKDIR "/scripts"
ENV SF_BACKEND=torch
RUN pip3 install slideflow[cucim,torch]==3.0.2 cupy-cuda11x slideflow-noncommercial slideflow-gpl versioneer && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/test.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/slideflow-studio.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/run_project.py && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/qupath_roi.groovy && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/3.0.2/scripts/qupath_roi_legacy.groovy && \
    wget https://raw.githubusercontent.com/slideflow/slideflow/master/scripts/welcome.py && \
    chmod +x /scripts/welcome.py

# Set entrypoint
ENTRYPOINT ["bash", "-c", "python3 /scripts/welcome.py && exec /bin/bash"]
