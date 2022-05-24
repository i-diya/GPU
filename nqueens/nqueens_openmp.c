#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#include <omp.h>

int N;
#define THREADS 16

bool stable(int queens[N], int n){
    for (int i = 0; i < n; i++){
        for(int j=i+1;j<n;j++){
            if (queens[i] == queens[j]) return false;
            if (queens[i] - queens[j] == i - j || queens[i] - queens[j] == j - i)
			    return false;
        }
    }
    return true;
}


int main (int argc, char **argv){

    int res=0;

     if (argc < 2) {
        printf ("Insufficient number of arguments\n");
        return 0;
    }

    int n = atoi(argv[1]);
    omp_set_num_threads(THREADS);
	N=n;
    long long int iter=pow(n,n);
    #pragma omp parallel for
    for (int i = 0; i < iter; i++){
        int curridx=i;
        int queens[n];
        for (int j = 0; j < n; j++)
		{
			queens[j] = curridx % n;
			
			curridx /= n;
		}
        if (stable(queens, n)){ 
        #pragma omp atomic
		    res++;
        }
    }

    printf("N=%d, solutions=%d\n\n", n, res);


}
