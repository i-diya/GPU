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
    bool *isSet = (bool *)malloc(m * sizeof(bool));
    for (int i=0;i<m;i++){
        for(int j=0;j<m;j++){
            int weight=0;
            if(i!=j){
            weight = (rand() % 10) + 1;
            adj_mat[index(i,j,m)]=weight;
            }
            else adj_mat[index(i,j,m)]=weight;
           // printf("%d ",weight);
        }
        //printf("\n");
    }

    for (int i = 0; i < m; i++){
        dist[i] = INF;
        isSet[i] = false;
    }

    dist[0]=0;
    for (int v=0; v< m-1; v++) {
        int m1=INF;
        int mindex;
        for(int x=0;x<m;x++){
            if(isSet[x]==false && dist[x]<=m1){
                m1=dist[x];
                mindex=x;
            }
        }
        isSet[mindex]=true;
        for (int r = 0; r < m; r++){
            if (!isSet[r] && adj_mat[index(mindex,r,m)] && dist[mindex] != INF && dist[mindex] + adj_mat[index(mindex,r,m)] < dist[r])
                dist[r] = dist[mindex] + adj_mat[index(mindex,r,m)];
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
