Commands to run FFT

gcc -std=c99 -o data gen-fft-data.c 
./data N (Please make sure that N is power of 2)

A file N.txt is created

gcc -std=c99 -lm fft fft.c
time ./fft N.txt

cuda1 was used to test the code
module load cuda-10.2
nvcc -o fft-cuda fft-cuda.cu
time ./fft-cuda N.txt

gcc -std=c99 -fopenmp -lm fft-openmp fft-openmp.c
time ./fft-openmp N.txt

Note: The output isn't printed because the max tested size was 8388608