#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#define index(i, j, N)  ((i)*(N)) + (j)

int main (int argc, char **argv) {

    if (argc < 3) {
        printf ("Insufficient number of arguments. Please try again.\n");
        return 0;
    }

    bool* h_graph;

    int n = atoi (argv[1]);
    int m = atoi (argv[2]);

    h_graph = (bool *)malloc(n * n * sizeof(bool));
    memset(h_graph, false, sizeof(bool) * n * n);

    FILE *fp;
    char filename[15];
    sprintf(filename, "%d_%d.txt", n, m);
    fp = fopen (filename, "w");

    if (fp == NULL) {
        printf ("Opening file failed. Please try again.\n");
        return 0;
    }

    srand (time(NULL));

    fprintf (fp, "%d\n", n);
    fprintf (fp, "%d\n", m);
    int s = rand() % n;
    fprintf (fp, "%d\n", s);

    for (int i = 0; i < m; i++) {
        int u = rand() % n;
        int v = rand() % n;

        while (u == v || h_graph[index(u,v,n)]) {
            u = rand() % n;
            v = rand() % n;
        }

        if (h_graph[index(u,v,n)]) {
            for (int i = 0; i < n; i++) {
                if (!h_graph[index(u,i,n)]) {
                    v = i;
                    break;
                }
            }
        }

        if (h_graph[index(u,v,n)]) {
            i = i - 1;
            continue;
        }

        fprintf (fp, "%d %d\n", u, v);
    }

    fclose(fp);
    free (h_graph);

    return 0;
}
