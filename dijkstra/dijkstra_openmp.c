#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <string.h>

#include <omp.h>

#define INF 1000000
#define THREADS 16
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
    bool *isSet = (bool *)malloc(m * sizeof(bool));
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

    for (int i = 0; i < m; i++){
        dist[i] = INF;
        isSet[i] = false;
    }
    dist[0]=0;

    int nextnode;
    for (int v=0; v< m-1; v++) {
        int m1=INF;
        int mindex=-1;
        int min_dist_thread, min_node_thread;
        omp_set_num_threads(THREADS);
        #pragma omp parallel private(min_dist_thread, min_node_thread) shared(dist, isSet)
        {

            min_dist_thread = m1;
            min_node_thread=mindex;
            #pragma omp barrier 
            #pragma omp for nowait 
            for (int x = 0; x < m; x++) { 
                if ((dist[x] < min_dist_thread) && (isSet[x] == false)) {
                    min_dist_thread = dist[x];
                    min_node_thread = x;
                }
            }
            #pragma omp critical
            {
                if (min_dist_thread < m1) {
                    m1 = min_dist_thread;
                    mindex = min_node_thread;
                }
            }
        }

        isSet[mindex]=true;
        omp_set_num_threads(THREADS);
        int update_dist;
        #pragma omp parallel shared(adj_mat,dist)
        {
            #pragma omp for private(update_dist,nextnode)
            for (nextnode = 0; nextnode < m; nextnode++) {
                update_dist = dist[mindex]+ adj_mat[index(mindex,nextnode,m)];
                if ((isSet[nextnode] != true) && (adj_mat[index(mindex,nextnode,m)] != 0) && (update_dist < dist[nextnode])){
                    dist[nextnode] = update_dist; 
                }
            }
            #pragma omp barrier
        } 
    }

    printf("the shortest path from src to each of the vertices is as follows\n");
    for (int i = 0; i < m; i++){
        if(dist[i]>INF){
            dist[i]=INF;
        }
        printf("vertex=%d distance=%d\n",i+1,dist[i]);
    }

    return 0;
}
