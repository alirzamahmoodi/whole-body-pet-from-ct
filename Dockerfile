FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgl1-mesa-glx libglib2.0-0 \
  curl sudo git wget htop \
  && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
ARG USER_NAME="alirzamahmoodi"
RUN adduser --disabled-password --gecos '' --shell /bin/bash ${USER_NAME}
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}
USER ${USER_NAME}
ENV HOME=/home/${USER_NAME}
RUN chmod 777 /home/${USER_NAME}

# Install Miniconda
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
 && bash Miniconda3-py37_4.8.2-Linux-x86_64.sh -p ~/miniconda -b \
 && rm Miniconda3-py37_4.8.2-Linux-x86_64.sh
ENV PATH=/home/${USER_NAME}/miniconda/bin:$PATH
# Create the conda environment
COPY environment.yml /home/${USER_NAME}/environment.yml
RUN conda env create -f /home/${USER_NAME}/environment.yml \
 && conda clean -ya
ENV CONDA_DEFAULT_ENV=mahmoodi-pix2pixhd
ENV CONDA_PREFIX=/home/${USER_NAME}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Install additional Python packages
RUN pip install dominate==2.4.0 \
    visdom==0.1.8

# Set the working directory
WORKDIR /home/${USER_NAME}

# Example build and run commands (uncomment to use)
# docker build -t ctpet .
# docker run -it --rm --gpus all --shm-size=192G --user $(id -u):$(id -g) --cpuset-cpus=20-29 \
# -v /*:/Code \
# --name CTPET ctpet:latest
