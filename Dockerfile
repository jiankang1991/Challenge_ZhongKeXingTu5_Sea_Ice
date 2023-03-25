FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /workspace
RUN python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102
WORKDIR /workspace
CMD ["python", "test.py"]