# Lloyd's algorithm implementation usign DPC++

Implementation closely following https://github.com/soda-inria/sklearn-numba-dpex to compare pefrormance
of `numba_dpex`-based implementation and DPC++ implementation.

DPC++ implementation can be compiled usign intel/llvm open-source DPC++ compiler targeting CUDA, or HIP backends.

This is a work-in-progress work.

## Building

```bash
CXX=icpx python setup.py develop -- -DDPCTL_MODULE_PATH=$(python -m dpctl --cmakedir)
```

## Running tests

```bash
python -m pytest -s test/
```