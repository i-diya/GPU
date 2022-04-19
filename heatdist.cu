/* 
 * This file contains the code for doing the heat distribution problem. 
 * You do not need to modify anything except starting  gpu_heat_dist() at the bottom
 * of this file.
 * In gpu_heat_dist() you can organize your data structure and the call to your
 * kernel(s), memory allocation, data movement, etc. 
 * 
 */

#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 


/* To index element (i,j) of a 2D array stored as 1D */
#define index(i, j, N)  ((i)*(N)) + (j)

/*****************************************************************/

// Function declarations: Feel free to add any functions you want.
void  seq_heat_dist(float *, unsigned int, unsigned int);
void  gpu_heat_dist(float *, unsigned int, unsigned int);

__global__ void gpu_kernel(float * playground, float * cudatemp, unsigned int N, unsigned int iterations)
{
  int i,j;
  //printf("INSIDE KERNEL \n");
  //printf("CUDA TEMP KERNEL %lf",cudatemp[index(1,1,N)]);
  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;
  
  int upper = N-1;
  
    
   if ((i < upper) && (i > 0) && (j > 0) && (j < upper)){
        
	cudatemp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
      
      
    }
  
			      
   			      
    /* Move new values into old values */ 
  //printf("CUDA TEMP KERNEL %lf",cudatemp[index(1,1,N)]);
}

/*****************************************************************/
/**** Do NOT CHANGE ANYTHING in main() function ******/

int main(int argc, char * argv[])
{
  unsigned int N; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  int iterations = 0;
  int i;
  
  /* The 2D array of points will be treated as 1D array of NxN elements */
  float * playground; 
  
  // to measure time taken by a specific part of the code 
  double time_taken;
  clock_t start, end;
  
  if(argc != 4)
  {
    fprintf(stderr, "usage: heatdist num  iterations  who\n");
    fprintf(stderr, "num = dimension of the square matrix (50 and up)\n");
    fprintf(stderr, "iterations = number of iterations till stopping (1 and up)\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: GPU execution\n");
    exit(1);
  }
  
  type_of_device = atoi(argv[3]);
  N = (unsigned int) atoi(argv[1]);
  iterations = (unsigned int) atoi(argv[2]);
   //printf("HELLO /n");
  
  /* Dynamically allocate NxN array of floats */
  playground = (float *)calloc(N*N, sizeof(float));
  if( !playground )
  {
   fprintf(stderr, " Cannot allocate the %u x %u array\n", N, N);
   exit(1);
  }
  
  /* Initialize it: calloc already initalized everything to 0 */
  // Edge elements  initialization
  for(i = 0; i < N; i++)
    playground[index(0,i,N)] = 100;
  for(i = 0; i < N-1; i++)
    playground[index(N-1,i,N)] = 150;
  //printf("HELLO /n");
  if( type_of_device == 0 ) // The CPU sequential version
  {  
    start = clock();
    seq_heat_dist(playground, N, iterations);
   // printf("playground %lf \n",playground[index(1,0,N)]) ;
  //   printf("playground %lf \n",playground[index(0,1,N)]) ;
  //   printf("playground %lf \n",playground[index(2,1,N)]) ;
  //   printf("playground %lf \n",playground[index(1,2,N)]) ;
  //  printf("playground %lf \n",playground[index(1,1,N)]) ;
    end = clock();
  }
  else  // The GPU version
  { //printf("HELLO /n");
     start = clock();
    // printf("playground %lf \n",playground[index(0,0,N)]) ;
     gpu_heat_dist(playground, N, iterations);
    // printf("playground %lf \n",playground[index(1,0,N)]) ;
   //  printf("playground %lf \n",playground[index(0,1,N)]) ;
   //  printf("playground %lf \n",playground[index(2,1,N)]) ;
    // printf("playground %lf \n",playground[index(1,2,N)]) ;
    // printf("playground %lf \n",playground[index(1,1,N)]) ;
     end = clock();    
  }
  
  
  time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken for %s is %lf\n", type_of_device == 0? "CPU" : "GPU", time_taken);
  
  free(playground);
  
  return 0;

}


/*****************  The CPU sequential version (DO NOT CHANGE THAT) **************/
void  seq_heat_dist(float * playground, unsigned int N, unsigned int iterations)
{
  // Loop indices
  int i, j, k;
  int upper = N-1;
  
  // number of bytes to be copied between array temp and array playground
  unsigned int num_bytes = 0;
  
  float * temp; 
  /* Dynamically allocate another array for temp values */
  /* Dynamically allocate NxN array of floats */
  temp = (float *)calloc(N*N, sizeof(float));
  if( !temp )
  {
   fprintf(stderr, " Cannot allocate temp %u x %u array\n", N, N);
   exit(1);
  }
  
  num_bytes = N*N*sizeof(float);
  
  /* Copy initial array in temp */
  memcpy((void *)temp, (void *) playground, num_bytes);
  
  for( k = 0; k < iterations; k++)
  {
    /* Calculate new values and store them in temp */
    for(i = 1; i < upper; i++)
      for(j = 1; j < upper; j++)
	temp[index(i,j,N)] = (playground[index(i-1,j,N)] + 
	                      playground[index(i+1,j,N)] + 
			      playground[index(i,j-1,N)] + 
			      playground[index(i,j+1,N)])/4.0;
  
			      
   			      
    /* Move new values into old values */ 
    memcpy((void *)playground, (void *) temp, num_bytes);
  }
  
}

/***************** The GPU version: Write your code here *********************/
/* This function can call one or more kernels if you want ********************/
void  gpu_heat_dist(float * playground,  unsigned int N, unsigned int iterations)
{
  
  int size = (N*N)*sizeof(float);
  float *play, *cudatemp;
  cudaError_t err = cudaSuccess;
  err = cudaMalloc((void **) &play,size) ;
  

    if (err != cudaSuccess)
    {
        fprintf(stderr, "SIZE TOO LARGE \n");
        exit(EXIT_FAILURE);
    }
  cudaMalloc((void **) &cudatemp,(N*N)*sizeof(float)) ;
  cudaMemcpy( play, playground, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy( cudatemp, playground, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid(ceil(N/8.0),ceil(N/8.0));
  dim3 dimBlock(8.0,8.0);
  
  
  for( int k = 0; k < iterations; k++)
  {
    //printf("ITERATION \n");
  
  gpu_kernel<<<dimGrid, dimBlock>>>(play, cudatemp, N , iterations);
  
  //cudaThreadSynchronize();
  cudaMemcpy(play, cudatemp, N * N * sizeof(float), cudaMemcpyDeviceToDevice);
  
  }
 // printf("ITERATIONS OVER  \n");
  cudaMemcpy(playground, play, N * N * sizeof(float), cudaMemcpyDeviceToHost);
 // printf("THIS OVER  \n");
  cudaFree(play);
  cudaFree(cudatemp);
  
  
  
}



