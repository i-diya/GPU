#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>
#include <math.h>
#include <cuda.h>
#include <cuComplex.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

__device__ uint32_t revBits(uint32_t val){

    val = ((val & 0xaaaaaaaa) >> 1) | ((val & 0x55555555) << 1);
    val = ((val & 0xcccccccc) >> 2) | ((val & 0x33333333) << 2);
    val = ((val & 0xf0f0f0f0) >> 4) | ((val & 0x0f0f0f0f) << 4);
    val = ((val & 0xff00ff00) >> 8) | ((val & 0x00ff00ff) << 8);
    return (val >> 16) | (val << 16);
}


__global__ void fft_kernel(cuFloatComplex* input, cuFloatComplex* output, uint32_t N, int logN){

    uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;

    uint32_t rev = revBits(2*index);
    rev = rev>>(32-logN);
    output[2*index] = input[rev]; 

    rev = revBits(2*index+1);
    rev = rev>>(32-logN);
    output[2*index+1] = input[rev]; 

    __syncthreads();

    for(int Findex = 1; Findex<=logN; Findex++){

        int size = 1<<Findex;
        int size1 = 1<<(Findex-1);

        int j = threadIdx.x / size1 * size;
        int k = threadIdx.x % size1;

        cuFloatComplex t1 = output[j+k];

        float real, imaginary;

        sincosf((float) -M_PI * k/size1, &imaginary, &real);
        cuFloatComplex twiddle = make_cuFloatComplex(real, imaginary);

        cuFloatComplex t2 = cuCmulf(twiddle, output[j+k+size1]);

        output[j+k] = cuCaddf(t1, t2);
        output[j+k+size1] = cuCsubf(t1, t2);

        __syncthreads();
    }
}


int main(int argc, char *argv[]){

    char* filename = argv[1];
    FILE* fp = fopen (filename, "r");
    if (fp == NULL) {
        printf ("Opening file failed. Please try again.\n");
        return 0;
    }

    uint32_t N;
    fscanf (fp, "%u\n", &N);

    cuFloatComplex* input = (cuFloatComplex*) malloc(N * sizeof(cuFloatComplex));
    cuFloatComplex* output = (cuFloatComplex*) malloc(N * sizeof(cuFloatComplex)); 
    
    for(uint32_t i = 0; i<N; i++){
        float a, b;
        fscanf (fp, "%f %f\n", &a, &b);
        input[i] = make_cuFloatComplex(a,b);
    }
    
    int logN = (int) log2f((float) N);

    cuFloatComplex* input_d;
    cuFloatComplex* output_d;

    cudaMalloc((void**)&input_d, N * sizeof(cuFloatComplex));
    cudaMalloc((void**)&output_d, N * sizeof(cuFloatComplex));

    cudaMemcpy(input_d, input, N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    
    int size = N/2;
    int block_size = min(size, 1024);
    dim3 block(block_size, 1);
    dim3 grid((size + block_size - 1) / block_size, 1);

    fft_kernel<<<grid, block>>>(input_d, output_d, N, logN);

    cudaMemcpy(output, output_d, N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    // for(int i=0; i<N; i++){
    //     printf("%f + i%f\n", cuCrealf(output[i]), cuCimagf(output[i]));
    // }

    cudaFree(input_d);
    cudaFree(output_d);

}