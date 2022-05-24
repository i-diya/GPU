#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <complex.h>
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

uint32_t revBits(uint32_t val){

    val = ((val & 0xaaaaaaaa) >> 1) | ((val & 0x55555555) << 1);
    val = ((val & 0xcccccccc) >> 2) | ((val & 0x33333333) << 2);
    val = ((val & 0xf0f0f0f0) >> 4) | ((val & 0x0f0f0f0f) << 4);
    val = ((val & 0xff00ff00) >> 8) | ((val & 0x00ff00ff) << 8);
    return (val >> 16) | (val << 16);
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

    float complex* input = (float complex*) malloc(N * sizeof(float complex));
    float complex* output = (float complex*) calloc(N, sizeof(float complex)); 
    
    for (uint32_t i = 0; i < N; i++){
        float a, b;
        fscanf (fp, "%f %f\n", &a, &b);
        input[i] = a + b*I;
    }
        
    
    int logN = (int) log2f((float) N);

    for(uint32_t i=0; i<N; i++){
        uint32_t rev = revBits(i);
        rev = rev>>(32-logN);
        output[i] = input[rev]; 
    }

    for(int Findex = 1; Findex<=logN; Findex++){

        int size = 1<<Findex;
        int size1 = 1<<(Findex-1);

        float complex twiddle = cexpf(-2.0I * M_PI / size);

        for(uint32_t j = 0; j<N; j = j + size){
            float complex twiddleFactor = 1;
            for(int k = 0; k<size1; k++){
                float complex t1 = output[j+k];
                float complex t2 = twiddleFactor * output[j+k+size1];

                output[j+k] = t1 + t2;
                output[j+k+size1] = t1 - t2;

                twiddleFactor = twiddleFactor*twiddle;
            }
        }        
    }

    // for(int i=0; i<N; i++){
    //     printf("%f + i%f\n", creal(output[i]), cimag(output[i]));
    // }

    fclose(fp);
}

