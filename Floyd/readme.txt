

TO COMPILE:
g++ -std=c++11 -o nqueen nqueensserial.cpp
g++ -std=c++11 -o floyd floyd.cpp
gcc -fopenmp -o nqueen nqueens_openmp.c
g++ -std=c++11 -fopenmp -o floyd floyd.cpp
nvcc -o nqueens nqueens.cu
nvcc -o floyd floyd.cu

TO RUN:
./floyd
./nqueens N (N = number of queens)

To change vertices in floyd change #define VERTICES (10) in line 9.

