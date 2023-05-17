# syntax=docker/dockerfile:1
ARG BASE_IMAGE=tensorflow/tensorflow:2.9.3-gpu

# Build vips from source
FROM $BASE_IMAGE AS build-vips

# Install necessary packages
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    libexpat1-dev \
    libglib2.0-dev \
    libgsf-1-dev \
    libjpeg-turbo8-dev \
    libopenjp2-tools \
    libopenslide-dev \
    libpng-dev \
    libtiff-dev \
    libtiff5-dev \
    pkg-config \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Build libvips 8.12 from source [slideflow requires 8.9+, latest deb in Ubuntu 18.04 is 8.4]
RUN mkdir libvips
WORKDIR "/libvips"
RUN wget https://github.com/libvips/libvips/releases/download/v8.12.2/vips-8.12.2.tar.gz && \
    tar zxf vips-8.12.2.tar.gz
WORKDIR "/libvips/vips-8.12.2"
RUN ./configure && make && make install


# Build patched pixman package
FROM $BASE_IMAGE AS build-pixman

RUN mkdir /scripts
WORKDIR /scripts
RUN sed -i '/^#\sdeb-src /s/^# *//' "/etc/apt/sources.list" && \
    apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    devscripts \
    wget && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/2.0.1/scripts/pixman_repair.sh && \
    chmod +x pixman_repair.sh && \
    ./pixman_repair.sh && \
    rm ./pixman_repair.sh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Build wheel for spams from source
FROM $BASE_IMAGE AS build-spams
RUN mkdir /build
WORKDIR /build
RUN apt update && \
    apt install -y --no-install-recommends \
    build-essential \
    liblapack-dev \
    libblas-dev \
    python3-venv && \
    pip3 install --no-cache --no-input build && \
    pip3 download --no-cache --no-input --no-deps --no-binary 'spams' -v spams && \
    tar xzvf spams-*.tar.gz && \
    rm spams-*.tar.gz && \
    python3 -m build --outdir /build --wheel spams-* && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# Collect built packages and libraries from above
FROM $BASE_IMAGE
COPY --from=build-vips /usr/local/bin/vips* /usr/bin/
COPY --from=build-vips /usr/local/lib/libvips.so* /lib/x86_64-linux-gnu/
COPY --from=build-pixman /opt/pixman/*.deb /

# Install patched pixman package
RUN dpkg -i /libpixman-*.deb && rm /libpixman-*.deb
# Install necessary packages
RUN apt update && \
    apt install -y --no-install-recommends \
    expat \
    openslide-tools \
    libopenjp2-tools \
    liblapack3 \
    libblas3 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libexpat1 \
    libglib2.0-0 \
    libgsf-1-114 \
    libjpeg-turbo8 \
    libopenslide0 \
    libpng16-16 \
    libtiff5 \
    wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build-spams /build/*.whl /
# Install slideflow & download scripts
RUN pip3 install --no-cache --no-input \
    slideflow[cucim]==2.0.1 \
    cupy-cuda11x \
    tensorflow_datasets \
    tensorflow_probability==0.17.* && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/2.0.1/scripts/test.py && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/2.0.1/scripts/run_project.py && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/2.0.1/scripts/qupath_roi.groovy && \
    wget https://raw.githubusercontent.com/jamesdolezal/slideflow/2.0.1/scripts/qupath_roi_legacy.groovy && \
    pip3 install --no-cache --no-input --force-reinstall --no-deps /spams*.whl && \
    rm /spams*.whl
