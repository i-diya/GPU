#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdbool.h>

#define index(i, j, N)  ((i)*(N)) + (j)


__global__ void dijkstra_help(int *adj_mat, int *result, bool* visited, int V) {

    for (int count = 0; count < V-1; count++)
    {        
        int min = INT_MAX, u;
        for (int v = 0; v < V; v++)
            if (visited[index(V,v,blockIdx.x)] == false && result[index(V,v,blockIdx.x)] <= min)
                min = result[index(V,v,blockIdx.x)], u = v;
  
        visited[index(V,u,blockIdx.x)] = true;
  
        for (int v = 0; v < V; v++) {
            if (!visited[index(V,v,blockIdx.x)] && adj_mat[index(u,v,V)] && result[index(V,u,blockIdx.x)] != INT_MAX
                && result[index(V,u,blockIdx.x)] + adj_mat[index(u,v,V)] < result[index(V,v,blockIdx.x)])
                result[index(V,v,blockIdx.x)] = result[index(V,u,blockIdx.x)] + adj_mat[index(u,v,V)];
        }
    }
}

int main(int argc, char **argv) {
    
    if (argc < 2) {
        printf ("Insufficient number of arguments\n");
        return 0;
    }

    int m;

    int len;

 
    m=atoi(argv[1]);
    len=m*m;

    int* adj_mat = (int *) malloc(len * sizeof(int));
    int* result = (int *) malloc(len * sizeof(int));

    bool *visited= (bool *) malloc(len * sizeof(bool));
    

    
    for (int i = 0; i < m; i++) { 
	    for(int j = 0; j < m; j++) {
            int weight=0;
            if(i!=j){
            weight = (rand() % 10) + 1;
            adj_mat[index(i,j,m)]=weight;
            }
            else adj_mat[index(i,j,m)]=weight;
        }
    }
    

    int *d_adj_mat, *d_result;
    bool *d_visited;

    cudaMalloc((void **) &d_adj_mat, (len * sizeof(int)));
    cudaMalloc((void **) &d_result, (len * sizeof(int)));
    cudaMalloc((void **) &d_visited, (len * sizeof(bool)));

    for(int i=0;i<len;i++){
        if(i%m==0)
        result[i]=0;
        else result[i]=INT_MAX;
        visited[i] = false;
    }

    cudaMemcpy(d_visited, visited, (len * sizeof(bool)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, (len * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adj_mat, adj_mat, (len * sizeof(int)), cudaMemcpyHostToDevice);
    
    dijkstra_help<<<m,1>>>(d_adj_mat,d_result, d_visited, m);
    cudaMemcpy(result, d_result, (len * sizeof(int)), cudaMemcpyDeviceToHost);
    

    for(int i = 0; i < m; i++) {
            printf(" Distance from source to %d is %d \n",i+1, result[i]);
    }
    printf("\n");

    cudaFree(d_adj_mat);
    cudaFree(d_result);
    cudaFree(d_visited);

    free(adj_mat);
    free(result);
    free(visited);


    return 0;

}