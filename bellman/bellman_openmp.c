#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#include <omp.h>

#define INF 1000000
#define index(i, j, N)  ((i)*(N)) + (j)

int *adj_mat;//adjacency matrix


int main (int argc, char **argv){

    if (argc < 2) {
        printf ("Insufficient number of arguments\n");
        return 0;
    }

    int m = atoi(argv[1]);

    adj_mat=(int *) malloc(m*m*sizeof(int));
    int *dist = (int *)malloc(m * sizeof(int));

    //store values in adjacency matrix
    for (int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            int weight=0;
            if(i!=j){
            weight = (rand() % 10) + 1;
            adj_mat[index(i,j,m)]=weight;
            }
            else adj_mat[index(i,j,m)]=weight;
            //printf("%d ",weight);
        }   
    }
    
    memset(dist, INF, sizeof(int) * m);
    dist[0] = 0;

    #pragma omp target enter data map(to: adj_mat[0:m], dist[0:m])
    for (int i = 0; i < m - 1; i++) {
        #pragma omp teams distribute parallel for collapse(2)
        for(int src=0;src<m;src++){
            for(int dst=0;dst<m;dst++){

                if (dist[src] != INF && (dist[dst] > dist[src] + adj_mat[index(src,dst,m)])){
                    dist[dst] = dist[src] + adj_mat[index(src,dst,m)];
                }
            }
        }

    }
    #pragma omp target exit data map(from: dist[0:m])

    free (adj_mat);
    printf("the shortest path from src to each of the vertices is as follows\n");
    for (int i = 0; i < m; i++){
        if(dist[i]>INF){
            dist[i]=INF;
        }
        printf("vertex=%d distance=%d\n",i+1,dist[i]);
    }

    return 0;
    

}
