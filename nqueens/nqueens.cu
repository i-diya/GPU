#include <cassert>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <stack>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sstream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include<time.h>

#define N_MAX 12

  double time_taken;
  clock_t start, end;

__global__ void gener(const int N, int long long O, int long long offset, int *d_solutions, int *d_num_solutions) {
    //long long int column = (long long int)getGlobalIdx_3D_3D() + offset; //use this line for the 3D-3D implementation
   int long long column = (int long long)(threadIdx.x + blockIdx.x * blockDim.x) + offset; //use this line for the 2D implementation
    if (column >= O)
        return;
    //vector<int> gameboard;
    bool boardIsValidSoFar;
    int gameBoard[N_MAX];

    for (int i = 0; i < N; i++) {
        gameBoard[i] = column % N;
        //printf(" GAMEBOARD %d ",gameBoard[i]);
        int lastPlacedRow = i;
        int lastPlacedColumn = gameBoard[lastPlacedRow];
        boardIsValidSoFar= true;
    for (int row = 0; row < lastPlacedRow; ++row)
    {
        if (gameBoard[row] == lastPlacedColumn)
            {boardIsValidSoFar= false;}
        int col1 = lastPlacedColumn - (lastPlacedRow - row);
        int col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            {boardIsValidSoFar= false;}
    }
    

    if (!boardIsValidSoFar)
            return;

        column /= N;
    }
    //atomicAdd(d_num_solutions, 1) ;
   // printf("NUM SOLUTINS: %d\n",&d_num_solutions) ;
    const int index = atomicAdd(d_num_solutions,1);
   // printf("NUM SOLUTINS: %d\n",&d_num_solutions) ;
   // for (int i = 0; i < N; i++)
     //   d_solutions[N * index + i] = gameBoard[i];
}


int long long factorial(int n)  
{  
  if (n == 0)  
    return 1;  
  else  
    return(n * factorial(n-1));  
} 






void generate(const int N)
{
    std::vector<std::vector<int> > solutions;
    int num_solutions = 0;

    //int *h_num_solutions = 0;
    int *d_solutions ;
    int *d_num_solutions;
    const long long int O = powl(N,N);
   // printf("here ?");
    //size_t solutions_mem = (factorial(N)) * sizeof(int*); //N^5 is an estimation of the amount of solutions for size N (^5 because N_MAX^4 (12^4) is enough to hold all the solutions for a 12x12 board and to store N columns for that board that would make it N^5)
    //cudaMalloc(&d_solutions, (factorial(N)) * sizeof(int));
    cudaMalloc(&d_num_solutions, sizeof(int));

    cudaMemcpy(d_num_solutions, &num_solutions, sizeof(int), cudaMemcpyHostToDevice);
    
    int id_offsets = 1; //initialise as 1 so that the kernel is executed at least once
    


    /* use these two lines with the 2D kernel implementation */
    int grid = 1024;
    int block = 512;
    if (O > grid * block)
        id_offsets = std::ceil((double)O / (grid * block));


   // printf("here");
    for (long long int i = 0; i < id_offsets; i++) {
  
        gener<<<grid, block>>>(N, O, (long long int)grid * block * i, d_solutions, d_num_solutions); //use this kernel invocation for the 2D implementation
        //printf(" i = %d ",i);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(&num_solutions, d_num_solutions, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_num_solutions);
    
    /*int* h_solutions = (int*)malloc((factorial(N)) * sizeof(int));
    cudaMemcpy(h_solutions, d_solutions, (factorial(N)) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_solutions);*/



    /* use this for loop with the array-based "gameBoard" implementation */
  /*  for (int i = 0; i < *h_num_solutions; i++) {
        std::vector<int> solution = std::vector<int>();
        for (int j = 0; j < N; j++)
            solution.push_back(h_solutions[N * i + j]);
        solutions.push_back(solution);
    }*/

   // free(h_solutions);

    printf("N=%d, solutions=%d\n\n", N, num_solutions);


}



int main(int argc, char** argv)
{
  //  printf("here");
  start = clock();
        int N=atoi(argv[1]);
       // printf("here ??");
       
        generate(N);
        end = clock();
        time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
  
  printf("Time taken is %lf\n", time_taken);
}