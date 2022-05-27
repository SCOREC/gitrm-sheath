
## layout

- src - source codes (will be installed with `make
install`)
- test - example code that exercises APIs
- CMakeLists.txt - cmake files that specify how to build the library and test

## setup

```
git clone https://github.com/SCOREC/gitrm-sheath.git
```

## build

The following assumes that a valid C and C++ compiler, and `cmake`, are in your PATH.

`CMAKE_INSTALL_PREFIX` is the path where the library, headers, and test binary
are installed.

`kk` is the path where kokkos is installed (either GPU or OpenMP based installation).

Set path as `export kk=/path/to/kokkos/install`

`kk_compiler` is the path to kokkos compiler
Set path as `export kk_compiler=/path/to/nvcc_wrapper` for GPU build.

Ignore this for OpenMP build

Load necessary modules:
```
module unuse /opt/scorec/spack/lmod/linux-rhel7-x86_64/Core
module use /opt/scorec/spack/v0154_2/lmod/linux-rhel7-x86_64/Core
module load \
gcc/10.1.0 \
openmpi \
cmake/3.20.0  \
cuda/11.4
```

Building with GPU
```
mkdir build-GPU
cd build-GPU
export CMAKE_PREFIX_PATH=$kk/lib64/cmake/Kokkos:$CMAKE_PREFIX_PATH
cmake ../gitrm-sheath -DCMAKE_CXX_COMPILER=$kk_compiler -DCMAKE_INSTALL_PREFIX=$PWD/install # on GPU
make -j 8
make install
```

Building with OpenMP
```
mkdir build-omp
cd build-omp
export CMAKE_PREFIX_PATH=$kk/lib64/cmake/Kokkos:$CMAKE_PREFIX_PATH
cmake ../gitrm-sheath -DCMAKE_INSTALL_PREFIX=$PWD/install # on OpenMP
make -j 8
make install
```
## test

```
./install/bin/GitrmSheath_Demo
```
