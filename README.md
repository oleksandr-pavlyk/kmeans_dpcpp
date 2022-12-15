# Lloyd's algorithm implementation using DPC++

Implementation closely following https://github.com/soda-inria/sklearn-numba-dpex to
compare performance of `numba_dpex`-based implementation and DPC++ implementation.

DPC++ implementation can be compiled using intel/llvm open-source DPC++ compiler
targeting CUDA, or HIP backends.

This is a work-in-progress work.

## Building


1. Create a Python 3.9 conda environment with `ninja` as a build system.

```shell
conda create -n env_name python=3.9 cmake ninja
```

2. Activate this environment.

```shell
conda activate env_name
```

3. Install packages from PyPi

```shell
pip install --no-cache-dir packaging setuptools distro scikit-build pytest numpy
```

4. Activate oneAPI compiler and TBB by exporting environment variables:

```shell
source /opt/intel/oneapi/compiler/latest/env/vars.sh
source /opt/intel/oneapi/tbb/latest/env/vars.sh
```

5. Install `dpctl`

   - Build it from source

        ```shell
        git clone https://github.com/IntelPython/dpctl
        pushd dpctl
        python scripts/build_locally.py
        popd
        ```

   - Alternatively you can also install `dpctl` from the `dppy/label/dev` conda channel,
   but you need to ensure that oneAPI compiler's version matches the version of the
   `dpcpp-cpp-rt` conda package installed as dependency of the `dpctl` conda package.

6. Install `kmeans_dpcpp` (this project) from source

```shell
git clone https://github.com/oleksandr-pavlyk/kmeans_dpcpp.git
cd kmeans_dpcpp
python setup.py develop -- -DCMAKE_CXX_COMPILER:PATH=$(which icpx) -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
```

Notice that `-DCMAKE_CXX_COMPILER` setting is necessary for `find_package(IntelDPCPP REQUIRED)` to work.

One can alternatively use `CXX` environment variable to specify the compiler:

```shell
CXX=icpx python setup.py develop -- -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
```

## Running tests

```bash
python -m pytest -s test/
```