#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <stdbool.h>
#include <string.h>


#define INF 1000000
#define index(i, j, N)  ((i)*(N)) + (j)

int *adj_mat;//adjacency matrix

int main (int argc, char **argv) {
    if (argc < 2) {
        printf ("Insufficient number of arguments\n");
        return 0;
    }

    int m = atoi(argv[1]);//number of vertices
    adj_mat=(int *) malloc(m*m*sizeof(int));
    int *dist = (int *)malloc(m * sizeof(int));

    for (int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            int weight=0;
            if(i!=j){
            weight = (rand() % 10) + 1;
            adj_mat[index(i,j,m)]=weight;
            }
            else adj_mat[index(i,j,m)]=weight;
        }
    }

    memset(dist, INF, sizeof(int) * m);
    dist[0] = 0;


    for(int i=0;i<m-1;i++){
        for(int src=0;src<m;src++){
            for(int dst=0;dst<m;dst++){

                if (dist[src] != INF && (dist[dst] > dist[src] + adj_mat[index(src,dst,m)])){
                    dist[dst] = dist[src] + adj_mat[index(src,dst,m)];
                }
            }
        }
    }

    free(adj_mat);
    //free(dist);


    printf("the shortest path from src to each of the vertices is as follows\n");
    for (int i = 0; i < m; i++){
        if(dist[i]>INF){
            dist[i]=INF;
        }
        printf("vertex=%d distance=%d\n",i+1,dist[i]);
    }

    return 0;

}