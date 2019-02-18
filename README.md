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
sudo apt install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
    cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
    libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0

#Optional: Install the TensorRT runtime (must be after CUDA install)
sudo apt update
sudo apt install libnvinfer4=4.1.2-1+cuda9.0

