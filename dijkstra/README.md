To compile the serial version of the Dijkstra algorithm, go to the dijkstra directory and execute the following:

gcc -std=c99 -lm serial.c

Now run the executable using:
./a.out V

where V is the number of vertices


To compile the OpenMP version of the Dijkstra algorithm, go to the dijkstra directory and execute the following:

gcc -fopenmp -Wall -std=c99 -o dijkstra_openmp dijkstra_openmp.c

Now run the executable using:
./dijkstra_openmp V


To compile the CUDA version of the Dijkstra algorithm, go to the dijkstra directory and execute the following:(I used the cuda1.cims.nyu.edu to run the program)

module load cuda-10.2

nvcc -o dijk dijk.cu

Now run the executable using:
./dijk V

