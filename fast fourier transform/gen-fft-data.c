#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

int main(int argc, char **argv){

    uint_fast32_t n = atoi (argv[1]);

    FILE *fp;
    char filename[15];
    sprintf(filename, "%d.txt", n);
    fp = fopen (filename, "w");

    if (fp == NULL) {
        printf ("Opening file failed. Please try again.\n");
        return 0;
    }

    srand (time(NULL));

    fprintf(fp, "%u\n", n);

    for(uint_fast32_t i = 0; i < n; i++){
        float a = (float) rand()/ (float) (RAND_MAX/20);
        float b = (float) rand()/ (float) (RAND_MAX/20);

        fprintf (fp, "%f %f\n", a, b);
    }

    fclose(fp);
    return 0;
}