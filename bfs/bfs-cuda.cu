#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define index(i, j, N)  ((i)*(N)) + (j)

__global__
void bfs (bool* d_graph, bool* d_q, bool* d_visited, int n, int u)
{
    int v = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (v < n) {
        if (d_graph[index(u,v,n)] && !d_visited[v]) {
            d_q[v] = true;
        }
    }
}


int main (int argc, char **argv) {

    if (argc < 2) {
        printf ("Insufficient number of arguments. Please try again.\n");
        return 0;
    }

    char* filename = argv[1];
    FILE* fp = fopen (filename, "r");
    if (fp == NULL) {
        printf ("Opening file failed. Please try again.\n");
        return 0;
    }

    bool* h_graph;
    bool* d_graph;

    int n, m, s;
    fscanf (fp, "%d\n", &n);
    fscanf (fp, "%d\n", &m);
    fscanf (fp, "%d\n", &s);

    h_graph = (bool *)malloc(n * n * sizeof(bool));
    memset(h_graph, false, sizeof(bool) * n * n);

    for (int i = 0; i < m; i++) {
        int u,v;
        fscanf (fp, "%d %d\n", &u, &v);

        h_graph[index(u,v,n)] = true;
    }

    bool* h_q;
    bool* d_q;
    h_q = (bool *) calloc (n, sizeof (bool));
    h_q[s] = true;

    bool* h_visited;
    bool* d_visited;
    h_visited = (bool*) calloc (n, sizeof(bool));

    cudaMalloc ((void**) &d_graph, n * n * sizeof (bool));
    cudaMemcpy (d_graph, h_graph, n * n * sizeof (bool), cudaMemcpyHostToDevice);

    cudaMalloc ((void**) &d_visited, n * sizeof(bool));

    cudaMalloc ((void**) &d_q, n * sizeof(bool));

    bool isBfsIncomplete = true;
    int count = 0;
    while (isBfsIncomplete) {
        count++;
        isBfsIncomplete = false;

        cudaMemcpy (d_visited, h_visited, n * sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemset (d_q, false, n * sizeof(bool));

        for (int i = 0; i < n; i++) {

            if (h_q[i] && !h_visited[i]) {
                isBfsIncomplete = true;
                h_visited[i] = true;

                printf ("%d\n", i);

                dim3 dimGrid((n/1024) + (n % 1024 == 0 ? 0 : 1));
                bfs<<<dimGrid, 1024>>> (d_graph, d_q, d_visited, n, i);
            }
        }

        cudaDeviceSynchronize();

        cudaMemcpy (h_q, d_q, n * sizeof(bool), cudaMemcpyDeviceToHost);
    }

    cudaFree (d_visited);
    cudaFree (d_graph);
    cudaFree (d_q);


    free (h_graph);
    free (h_visited);
    free (h_q);

    fclose(fp);

    return 0;
}
