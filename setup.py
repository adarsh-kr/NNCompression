import os, sys
from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11']#, '-stdlib=libc++', '-mmacosx-version-min=10.7']
cpp_link_args = [ '-lavcodec', '-lavutil', '-lavformat', '-lswscale']

ext_modules = [
    Extension(
    'wrap',
        ['funcs.cpp', 'wrap.cc', 'FeatureMap.cc', 'Codecs.cc'],
        include_dirs=['/usr/local/include/python3.6', '/usr/include/python3.6m', '/home/adarsh/.local/include/python3.6m'],
        language='c++',
        extra_compile_args = cpp_args,
        extra_link_args = cpp_link_args,
    ),
]

setup(
    name='wrap',
    version='0.0.1',
    author='adarsh-kr',
    author_email='adarsh@cs.wisc.edu',
    description='Compression',
    ext_modules=ext_modules,
)
