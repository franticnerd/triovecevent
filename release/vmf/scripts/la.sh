cd ..
make build
java -cp bin:lib/* vmf/onlineVmfGaussianMixture ../../data/la/ 6 600 50 50 1 1 0.01 0.01 1 3 3000 1 0 1 0.5 2.0 0.01

