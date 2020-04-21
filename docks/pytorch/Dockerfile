FROM nvidia/cuda:8.0-runtime-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# Set home
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create py env
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 8.0-specific steps
RUN conda install -y -c pytorch \
    cuda80=1.0 \
    magma-cuda80=2.3.0 \
    "pytorch=1.0.0=py3.6_cuda8.0.61_cudnn7.1.2_1" \
    torchvision=0.2.1 \
 && conda clean -ya

# Install hdf5 bindings
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

# Install torchnet
RUN pip install torchnet==0.0.4

# Install requests
RUN conda install -y requests=2.19.1 \
 && conda clean -ya

# Install opencv3 bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
RUN conda install -y -c menpo opencv3=3.1.0 \
 && conda clean -ya

## RICK -- install postgres tools
#RUN sudo apt-get update
#RUN sudo apt-get install -y curl
#RUN sudo apt-get install -y libpq-dev

# Added requirements file
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install jupyter notebook
RUN conda install -c conda-forge jupyterlab

# Install graphviz later
RUN conda install graphviz

# Set python path
ENV PYTHONPATH=$PYTHONPATH:/usr/src/app

# Set working directory to where i want
WORKDIR /usr/src/app

# Set the default command to python3
CMD ["python3"]