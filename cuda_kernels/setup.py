from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_kernels',
    ext_modules=[
        CUDAExtension('cuda_kernels', [
            'simple_kernels.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                          'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
