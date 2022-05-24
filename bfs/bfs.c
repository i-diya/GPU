#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define index(i, j, N)  ((i)*(N)) + (j)

void bfs (bool* h_graph, int u, bool* h_q_2, int n, bool* h_visited) {
    for (int i = 0; i < n; i++) {
        if (h_graph[index(u,i,n)] && !h_visited[i]) {
            h_q_2[i] = true;
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

    bool isBfsIncomplete = true;
    int count = 0;
    while (isBfsIncomplete) {
        count++;
        isBfsIncomplete = false;

        for (int i = 0; i < n; i++) {

            if (h_q[i] && !h_visited[i]) {
                isBfsIncomplete = true;
                h_visited[i] = true;

                printf ("%d\n", i);

                bfs (h_graph, i, h_q_2, n, h_visited);
            }
        }

        bool* temp = h_q;
        h_q = h_q_2;
        h_q_2 = temp;
    }


    free (h_graph);
    free (h_visited);
    free (h_q);
    free (h_q_2);

    fclose(fp);

    return 0;
}
