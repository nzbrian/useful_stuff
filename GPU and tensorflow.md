# useful_stuff

# VMWare ESXi: Passing Through an NVidia graphics card to a VM
https://ianmcdowell.net/blog/esxi-nvidia/
hypervisor.cpuid.v0 = FALSE

# NVIDIA GPU PPA
https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa
sudo add-apt-repository ppa:graphics-drivers/ppa


# CUDA on Ubuntu 18.04
based on https://www.tensorflow.org/install/gpu

#Add NVIDIA package repository
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt install ./cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt update

#Install CUDA and tools. Include optional NCCL 2.x
sudo apt install cuda-10-0 cuda-cublas-10-0 cuda-cufft-10-0 cuda-curand-10-0 \
    cuda-cusolver-10-0 cuda-cusparse-10-0 libcudnn7 libcudnn7-dev \
    libnccl2 libnccl-dev cuda-command-line-tools-10-0 \
    cuda-cublas-dev-10-0 cuda-cufft-dev-10-0 cuda-curand-dev-10-0 cuda-cusolver-dev-10-0 cuda-cusparse-dev-10-0

#download and install tensorrt
#don't do this https://developer.nvidia.com/nvidia-tensorrt-download
#do this:
sudo apt install python-libnvinfer python-libnvinfer-dev

# Build Tensorflow on Python3
based on https://www.tensorflow.org/install/source

sudo apt install python3-dev python3-pip
sudo python3 -m pip uninstall pip
sudo apt install python3-pip --reinstall

pip3 install -U --user six numpy wheel mock
pip3 install -U --user keras_applications==1.0.6 --no-deps
pip install -U --user keras_preprocessing==1.0.5 --no-deps

#to avoid error at end not finding keras_applications, also do
#pip3 install keras_applications==1.0.4 --no-deps
#pip3 install keras_preprocessing==1.0.2 --no-deps
#pip3 install h5py==2.8.0

#install Bazel - the build tool
sudo apt install pkg-config zip g++ zlib1g-dev unzip
#download binary installer from https://github.com/bazelbuild/bazel/releases   looks like:
https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
chmod +x bazel-*
./bazel-* --user

add this to the end of your ~/.bashrc   export PATH="$PATH:$HOME/bin"

export PATH="$PATH:$HOME/bin"

#clone the repo
git clone https://github.com/tensorflow/tensorflow.git

#check which capabilities your GPU has
https://developer.nvidia.com/cuda-gpus. (1050 has 6.1)

#build
cd tensorflow
./configure


Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: 
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
Do you wish to build TensorFlow with ROCm support? [y/N]: 
Do you wish to build TensorFlow with CUDA support? [y/N]: y
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10.0]:
Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Do you wish to build TensorFlow with TensorRT support? [y/N]: y
Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:
Please specify the locally installed NCCL version you want to use. [Default is to use https://github.com/nvidia/nccl]:

Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]:6.1

Do you want to use clang as CUDA compiler? [y/N]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Do you wish to build TensorFlow with MPI support? [y/N]:
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 
Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 


bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#install
pip3 install /tmp/tensorflow_pkg/tensorflow-version-tags.whl


# Build Tensorflow on Python2
sudo apt install python-dev python-pip
sudo python -m pip uninstall pip
sudo apt install python-pip --reinstall

pip install -U --user six numpy wheel mock
pip install -U --user keras_applications==1.0.6 --no-deps
pip install -U --user keras_preprocessing==1.0.5 --no-deps


#to avoid error at end not finding keras_applications, also do
#pip install keras_applications==1.0.4 --no-deps
#pip install keras_preprocessing==1.0.2 --no-deps
#pip install h5py==2.8.0

#install Bazel - the build tool
sudo apt install pkg-config zip g++ zlib1g-dev unzip
#download binary installer from https://github.com/bazelbuild/bazel/releases   looks like:
https://github.com/bazelbuild/bazel/releases/download/0.22.0/bazel-0.22.0-installer-linux-x86_64.sh
chmod +x bazel-*
./bazel-* --user

add this to the end of your ~/.bashrc   export PATH="$PATH:$HOME/bin"

export PATH="$PATH:$HOME/bin"

#clone the repo
git clone https://github.com/tensorflow/tensorflow.git

#check which capabilities your GPU has
https://developer.nvidia.com/cuda-gpus. (1050 has 6.1)

#build
cd tensorflow
./configure


Please specify the location of python. [Default is /usr/bin/python]:
Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
Do you wish to build TensorFlow with XLA JIT support? [Y/n]: 
Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
Do you wish to build TensorFlow with ROCm support? [y/N]: 
Do you wish to build TensorFlow with CUDA support? [y/N]: y
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 10.0]:
Please specify the location where CUDA 10.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7]: 
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Do you wish to build TensorFlow with TensorRT support? [y/N]: y
Please specify the location where TensorRT is installed. [Default is /usr/lib/x86_64-linux-gnu]:
Please specify the locally installed NCCL version you want to use. [Default is to use https://github.com/nvidia/nccl]:

Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 3.5,7.0]:6.1

Do you want to use clang as CUDA compiler? [y/N]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Do you wish to build TensorFlow with MPI support? [y/N]:
Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 
Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: 


bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package

./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

#install
pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
