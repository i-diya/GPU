#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdbool.h>

#define INF 1000000
#define index(i, j, N)  ((i)*(N)) + (j)

__global__
void bellman_ford_help (int* cuda_adj_mat, int* cuda_distance, int m){
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int update=blockDim.x * gridDim.x;

    if(idx>=m)
    return;

    for(int src = 0 ; src < m ; src++){
        for(int dst=idx;dst<m;dst+=update){
            if (cuda_distance[src] != INF && (cuda_distance[dst] > cuda_distance[src] + cuda_adj_mat[index(src,dst,m)])){
                cuda_distance[dst] = cuda_distance[src] + cuda_adj_mat[index(src,dst,m)];
            }
        }
    }

}

int main (int argc, char **argv){

    if (argc < 2) {
        printf ("Insufficient number of arguments\n");
        return 0;
    }

    int m = atoi(argv[1]);//number of vertices
    int *adj_mat;//adjacency matrix
    adj_mat=(int *) malloc(m*m*sizeof(int));

    int *cuda_adj_mat;
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

    int *cuda_distance;

    cudaMalloc ((void**) &cuda_adj_mat, m*m * sizeof(int));
    cudaMemcpy (cuda_adj_mat, adj_mat, m*m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc ((void**) &cuda_distance, m * sizeof(int));
    cudaMemcpy (cuda_distance, dist, m * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < m - 1; i++) {
        dim3 dimGrid(((m*m)/1024) + ((m*m) % 1024 == 0 ? 0 : 1));
        bellman_ford_help<<<dimGrid, 1024>>>(cuda_adj_mat, cuda_distance,m);
        cudaDeviceSynchronize();
    }

    cudaMemcpy (dist, cuda_distance, m * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree (cuda_adj_mat);
    cudaFree (cuda_distance);

    for (int i = 0; i < m; i++) {
        printf("%d : %d\n", i+1, dist[i]);
    }
    free (adj_mat);
    free (dist);

    return 0;
}
