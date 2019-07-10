# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV http_proxy=10.0.3.12:8118
ENV https_proxy=10.0.3.12:8118

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get -o Acquire::http::proxy="http://10.0.3.12:8118" update
# ==================================================================
# tools
# ------------------------------------------------------------------
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -o Acquire::http::proxy="http://10.0.3.12:8118" \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev
# ==================================================================
# python
# ------------------------------------------------------------------
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -o Acquire::http::proxy="http://10.0.3.12:8118" software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -o Acquire::http::proxy="http://10.0.3.12:8118" update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -o Acquire::http::proxy="http://10.0.3.12:8118" \
        python3.6 \
        python3.6-dev

RUN wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python

RUN pip3 install setuptools

RUN pip3 install \
        numpy \
        scipy \
        matplotlib \
        Cython

# ==================================================================
# pytorch
# ------------------------------------------------------------------
RUN pip3 install \
        torch_nightly -f \
        https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html \
        && \
    pip3 install \
        torchvision_nightly
    #$PIP_INSTALL \
    #     https://download.pytorch.org/whl/cu80/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
    #     && \
    #$PIP_INSTALL \
    #     torchvision \
    #     && \

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    pip3 install \
        shapely fire pybind11 tensorboardX protobuf \
        scikit-image pillow easydict ipython
RUN pip3 install numba==0.41.0
#  OPENCV
#RUN apt-get install python-opencv
RUN pip3 install opencv-python
RUN apt-get -o Acquire::http::proxy="http://10.0.3.12:8118" update
RUN apt-get -o Acquire::http::proxy="http://10.0.3.12:8118" install -y libsm6 libxext6 libxrender1 libfontconfig1

WORKDIR /root

RUN git clone https://github.com/open-mmlab/mmdetection.git
RUN cd ./mmdetection
RUN ./compile.sh
RUN python setup.py develop



