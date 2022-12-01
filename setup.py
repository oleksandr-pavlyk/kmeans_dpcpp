import skbuild

skbuild.setup(
    name="kmeans_dpcpp",
    version="0.0.1",
    description="Implementation of K-Means algorithm from sklearn-numba-dpex in SYCL using Intel DPC++ compiler",
    author="Intel Scripting",
    packages=["kmeans_dpcpp"],
)
