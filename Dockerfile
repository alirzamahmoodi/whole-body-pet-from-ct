FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install basic system dependencies
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
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -p ~/miniconda -b \
 && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/home/${USER_NAME}/miniconda/bin:$PATH

# Copy the updated environment file
COPY environment.yml /home/${USER_NAME}/environment.yml

# Create the conda environment
RUN conda env create -f /home/${USER_NAME}/environment.yml \
 && conda clean -ya

# Set up the conda environment
ENV CONDA_DEFAULT_ENV=mahmoodipix2pixhd
ENV CONDA_PREFIX=/home/${USER_NAME}/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# Set the working directory
WORKDIR /home/${USER_NAME}

# Install additional Python packages if needed
RUN pip install dominate==2.9.1 \
    visdom==0.2.4

# Optional: Uncomment to add commands for building and running the container
# docker build -t ctpet .
# docker run -it --rm --gpus all --shm-size=192G --user $(id -u):$(id -g) --cpuset-cpus=20-29 \
# -v /*:/Code \
# --name CTPET ctpet:latest
