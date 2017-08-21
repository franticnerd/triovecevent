cd ..
make build
mkdir -p ../../data/toy/cluster
java -cp bin:lib/* vmf/onlineVmfGaussianMixture ../../data/toy/ 6 200 50 10 1 1 0.01 0.01 1 3 3000 1 0 1 0.5 2.0 0.01

