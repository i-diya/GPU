Commands to run BFS

gcc -std=c99 -o data gen-bfs-data.c 
./data N M     (where N is the number of vertices and M is the number of edges)

A file N-M.txt is created

gcc -std=c99 -lm bfs bfs.c
time ./bfs N-M.txt

cuda1 was used to test the code
module load cuda-10.2
nvcc -o bfs-cuda bfs-cuda.cu
time ./bfs-cuda N-M.txt

gcc -std=c99 -fopenmp -lm bfs-openmp bfs-openmp.c
time ./bfs-openmp N-M.txt

Note: The output isn't printed because the max tested size was 10000 vertices and 2000000 edges.