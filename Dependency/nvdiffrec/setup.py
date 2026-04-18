import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

IS_HIP = hasattr(torch.version, 'hip') and torch.version.hip is not None

c_flags = ['-DNVDR_TORCH']
nvcc_flags = ['-DNVDR_TORCH']
ld_flags = []

if IS_HIP:
    c_flags += ['-D__HIP_PLATFORM_AMD__']
    nvcc_flags += ['-D__HIP_PLATFORM_AMD__']
    # No -lcuda -lnvrtc on ROCm
else:
    if os.name == 'posix':
        ld_flags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ld_flags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''

if IS_HIP:
    sources = [
        'nvdiffrec_render/renderutils/c_src/mesh.hip',
        'nvdiffrec_render/renderutils/c_src/loss.hip',
        'nvdiffrec_render/renderutils/c_src/bsdf.hip',
        'nvdiffrec_render/renderutils/c_src/normal.hip',
        'nvdiffrec_render/renderutils/c_src/cubemap.hip',
        'nvdiffrec_render/renderutils/c_src/common.cpp',
        'nvdiffrec_render/renderutils/c_src/torch_bindings.cpp',
    ]
else:
    sources = [
        'nvdiffrec_render/renderutils/c_src/mesh.cu',
        'nvdiffrec_render/renderutils/c_src/loss.cu',
        'nvdiffrec_render/renderutils/c_src/bsdf.cu',
        'nvdiffrec_render/renderutils/c_src/normal.cu',
        'nvdiffrec_render/renderutils/c_src/cubemap.cu',
        'nvdiffrec_render/renderutils/c_src/common.cpp',
        'nvdiffrec_render/renderutils/c_src/torch_bindings.cpp',
    ]

setup(
    name='nvdiffrec_render',
    packages=[
        'nvdiffrec_render',
        'nvdiffrec_render.renderutils',
    ],
    package_data={
        'nvdiffrec_render': ['bsdf_256_256.bin']
    },
    ext_modules=[
        CUDAExtension(
            name='nvdiffrec_render.renderutils._C',
            sources=sources,
            extra_compile_args={
                'cxx': c_flags,
                'nvcc': nvcc_flags,
            },
            extra_link_args=ld_flags,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
