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

#Optional: Install the TensorRT runtime (must be after CUDA install)
sudo apt update
sudo apt install libnvinfer4=4.1.2-1+cuda9.0

