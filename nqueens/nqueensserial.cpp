#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
//#include <stack>
#include <sstream>
#include <cmath>
//#include<atomic>
////#include<thread>

#define N_MAX 12


inline bool validity(int lastPlacedRow, const int* gameBoard, const int N)
{
    int lastPlacedColumn = gameBoard[lastPlacedRow];

    /* use this boolean when the below for loop is a Parallel For
    volatile bool valid = true;*/

//#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(guided) shared(valid)
    for (int row = 0; row < lastPlacedRow; ++row)
    {
        /* use this condition when this for is parallel
        if (!valid)
            continue;*/

        if (gameBoard[row] == lastPlacedColumn) // same column, fail!
            /* use this, and the following, returns when this for is not parallel (other wise use the following uses of the "valid" variable) */
            return false;
            //valid = false;
        // check the 2 diagonals
        const auto col1 = lastPlacedColumn - (lastPlacedRow - row);
        const auto col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            return false;
            //valid = false;
    }
    return true;
    //return valid;
}

int long long factorial(int n)  
{  
  if (n == 0)  
    return 1;  
  else  
    return(n * factorial(n-1));  
} 


void gener(const int N) {
    
    const long long int O = powl(N,N);
    int num_solutions;
 num_solutions=0;
 
       
bool here=true;
//omp_set_dynamic(0);
    //#pragma omp parallel for num_threads(std::thread::hardware_concurrency()) schedule(static)
     for (long long int i = 0; i < O; i++) {
         //num_solutions=0;
          bool valid = true;
        int gameBoard[N_MAX]; // OpenMP's performance improves drastically when using an array instead of a vector as the compiler will have it stored in the registers
        //std::vector<int> gameBoard(N, 0); // vector implementation of "gameBoard" - always runs slower than an array

        long long int column = i;

    for (int j = 0; j < N; j++) {
        gameBoard[j] = column % N;

        //printf(" GAMEBOARD %d ",gameBoard[i]);
   /*     int lastPlacedRow = j;
        int lastPlacedColumn = gameBoard[lastPlacedRow];
        here= true;
    for (int row = 0; row < lastPlacedRow; ++row)
    {
        if (gameBoard[row] == lastPlacedColumn)
            {here= false;}
        int col1 = lastPlacedColumn - (lastPlacedRow - row);
        int col2 = lastPlacedColumn + (lastPlacedRow - row);
        if (gameBoard[row] == col1 || gameBoard[row] == col2)
            {here= false;}
    }
    */
    

    if (!validity(j, gameBoard, N))
     { 
         valid=false;  
       // #pragma cancel for
         break;
     }

        column /= N;
        }
       // #pragma omp cancellation point for
    
     if (valid) {
        // printf(" HERE VALID ");
           // printf(" %d ",num_solutions);
            num_solutions++;
        }
         


}

int x = num_solutions;
printf("N=%d, solutions=%d\n\n", N, x);

}











int main(int argc, char** argv)
{
   // printf("here");
        int N=atoi(argv[1]);
       // printf("here ??");
         gener(N);
}

//g++ -o gfg -fopenmp t.cpp