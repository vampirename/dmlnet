"""
@Author: yfh
@Date:   2022/06/26
"""


import os.path as osp
import subprocess
import sys
from collections import defaultdict

import cv2
import torch


def is_rocm_pytorch() -> bool:

    is_rocm = False
    try:
        from torch.utils.cpp_extension import ROCM_HOME
        is_rocm = True if ((torch.version.hip is not None) and
                           (ROCM_HOME is not None)) else False
    except ImportError:
        pass

    return is_rocm

def _get_cuda_home():

    if is_rocm_pytorch():
        from torch.utils.cpp_extension import ROCM_HOME
        CUDA_HOME = ROCM_HOME
    else:
        from torch.utils.cpp_extension import CUDA_HOME

    return CUDA_HOME


def get_build_config():

    return torch.__config__.show()





def collect_base_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CPU: CPU information.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cpu = subprocess.check_output('cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c', shell=True)
    cpu = cpu.decode('utf-8').strip()
    env_info['CPU'] = cpu

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        CUDA_HOME = _get_cuda_home()
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    return env_info




def collect_env():
    """Collect the information of the running environments.
    
    You can add additional environment information by adding code like this:
    env_info['env_name'] = xxx._version__

    """
    env_info = collect_base_env()
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
