#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>

#define index(i, j, N)  ((i)*(N)) + (j)

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
    bool* h_q_2;
    h_q = (bool *) calloc (n, sizeof (bool));
    h_q_2 = (bool *) calloc (n, sizeof (bool));
    h_q[s] = true;

    bool* h_visited;
    h_visited = (bool*) calloc (n, sizeof(bool));

    #pragma omp target enter data map(to: h_graph[0:n * n], h_visited[0:n]), map(alloc:h_q_2[0:n])
    bool isBfsIncomplete = true;
    int count = 0;
    while (isBfsIncomplete) {
        count++;
        isBfsIncomplete = false;

        #pragma omp update to(h_q_2[0:n])
        #pragma omp update to(h_visited[0:n])
        for (int i = 0; i < n; i++) {

            if (h_q[i] && !h_visited[i]) {
                isBfsIncomplete = true;
                h_visited[i] = true;

                printf ("%d\n", i);



                #pragma omp teams distribute parallel for
                for (int j = 0; j < n; j++) {
                    if (h_graph[index(i,j,n)] && !h_visited[j]) {
                        h_q_2[j] = true;
                    }
                }
            }
        }

        #pragma omp update from(h_q_2[0:n])

        bool* temp = h_q;
        h_q = h_q_2;
        h_q_2 = temp;

        for (int j = 0; j < n; j++) {
            h_q_2[j] = false;
        }
    }


    free (h_graph);
    free (h_visited);
    free (h_q);
    free (h_q_2);

    fclose(fp);

    return 0;
}
