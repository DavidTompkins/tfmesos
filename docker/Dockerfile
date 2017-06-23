FROM nvidia/cuda:8.0-cudnn5-runtime

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libfreetype6-dev \
    libpng12-dev \
    libzmq3-dev \
    pkg-config \
    python \
    python-dev \
    rsync \
    software-properties-common \
    unzip \
    gcc \
    libopenblas-base \
    libzookeeper-mt-dev \
    ca-certificates \
    dnsutils \
    vim \
    git-core && \
    echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" > /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y --no-install-recommends bazel && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -c 'import urllib2;exec(urllib2.urlopen("https://bootstrap.pypa.io/get-pip.py").read())' --no-cache-dir --timeout 1000 && \
    pip install --no-cache-dir --timeout 1000 -r "https://raw.githubusercontent.com/DavidTompkins/tfmesos/master/requirements.txt" && \
    pip install --no-cache-dir --timeout 1000 "git+https://github.com/DavidTompkins/tfmesos.git@master#egg=tfmesos" && \
    git clone https://github.com/tensorflow/models.git && \
    cd /models/inception && \
    bazel build //inception:imagenet_distributed_train && \
    bazel build //inception:inception_distributed_train && \
    apt-get purge -y --auto-remove bazel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY examples /tmp

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# TensorBoard
EXPOSE 6006

ENV DOCKER_IMAGE dtompkins/tfmesos
