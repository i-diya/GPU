#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>


#define VERTICES (500)           //number of vertices for graph
#define MIN_EDGES_VERTEX (20)     //minimum no. of edges for each vertex
#define INF_DIST (10000000)       //Initial "infinite" distance value for each node
#define MAX_DIST (1000)  
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}      
int BLOCKSIZE=512;           


void Random(int* Graph, int* Edges_Per_Vertex);


void Initialize_Graph(int* Graph,int Value);


void Arr(int* Input_Array,int Value);

void Serial_Floyd(int* Host_Graph,int *Host_Path);


__global__ void CUDA_Kernel(int i,int* cuda_Graph, int* cuda_Path);


double timetaken();


struct timeval initial, final;




void Serial_Floyd(int *Graph1,int *Graph_Path){
    int x,y,z;
    for(x=0;x<VERTICES;++x){
        for(y=0;y<VERTICES;++y){
            for(z=0;z<VERTICES;++z){
		        int current_node=y*VERTICES+z;
		        int Node_i=y*VERTICES+x;
		        int Node_j=x*VERTICES+z;
               // printf("graph current node %d\n",Graph1[current_node]);
                if(Graph1[current_node]>(Graph1[Node_i]+Graph1[Node_j])){
                  //  printf("here change");
                    Graph1[current_node]=(Graph1[Node_i]+Graph1[Node_j]);
                    Graph_Path[current_node]=x;
                }
            }
        }
    }
     // for(int i=0;i<VERTICES*VERTICES;i++){
       //printf("graph in serial %d\n",Graph1[i]);
  // }
    //return Graph1;
}

/*This function computes shortest distance between all nodes parallely*/
__global__ void CUDA_Kernel(int i,int* cuda_Graph, int* cuda_Path){


   int t= (blockDim.x*blockDim.y)*threadIdx.z+    (threadIdx.y*blockDim.x)+(threadIdx.x); 
   

   int b= (gridDim.x*gridDim.y)*blockIdx.z+(blockIdx.y*gridDim.x)+(blockIdx.x);

   int T= blockDim.x*blockDim.y*blockDim.z;
   

   int B= gridDim.x*gridDim.y*gridDim.z;

for (int i1=b; i1<VERTICES; i1+=B)
   {
      for(int j=t; j<VERTICES; j+=T)
      {
         t=cuda_Graph[i1*VERTICES+i]+cuda_Graph[i*VERTICES+j];
         if(t<cuda_Graph[i1*VERTICES+j]){
         cuda_Graph[i1*VERTICES+j]=t;}
      }
   }
}


double timetaken(){
    double initial_s,final_s;
    double diff_s;
    initial_s= (double)initial.tv_sec*1000000 + (double)initial.tv_usec;
    final_s= (double)final.tv_sec*1000000 + (double)final.tv_usec;
    diff_s=(final_s-initial_s)/1000000;
    return diff_s;
}



int main(int argc, char** argv){
   

   
   printf("\nRunning Floyd Warshall's Algorithm");
   srand(8121);


   int Graph_Size=VERTICES*VERTICES*sizeof(int*);
   int *Graph=(int *)malloc(Graph_Size);
   int *Host_Graph=(int *)malloc(Graph_Size);
   int *Host_Path=(int *)malloc(Graph_Size);
   int *Device_Graph=(int *)malloc(Graph_Size);
   int *Device_Path=(int *)malloc(Graph_Size);
   int* Edges_Per_Vertex = (int*)malloc(VERTICES*sizeof(int));
   printf("\nBlocksize :%d",BLOCKSIZE);
   printf("\nVertices  :%d",VERTICES);


   Initialize_Graph(Graph,(int)0);
   for(int i=0;i<VERTICES;i++){
       Edges_Per_Vertex[i]=0;
   }

   Random(Graph,Edges_Per_Vertex);


   free(Edges_Per_Vertex);


   int i;
   for(i=0;i<VERTICES*VERTICES;i++){
       Host_Graph[i]=Graph[i];
       Host_Path[i]=-1;

   }
   printf("\nPerforming CPU computation");

   gettimeofday(&initial,NULL);


   Serial_Floyd(Host_Graph,Host_Path);

   gettimeofday(&final,NULL);
   double diff=0;


   diff=timetaken();

   printf("\nTime taken for logic computation by CPU in seconds is %f",diff);


   for(i=0;i<VERTICES*VERTICES;i++){
       Device_Graph[i]=Graph[i];
       Device_Path[i]=-1;
   }
   

   int* cuda_Graph;
   int* cuda_Path;
  

   cudaMalloc((void**)&cuda_Graph,Graph_Size);
   cudaMalloc((void**)&cuda_Path,Graph_Size);

   gettimeofday(&initial,NULL);

   cudaMemcpy(cuda_Graph, Device_Graph, Graph_Size, cudaMemcpyHostToDevice);
   cudaMemcpy(cuda_Path, Device_Path, Graph_Size,cudaMemcpyHostToDevice);
    

   gettimeofday(&final,NULL);

   double diff2=0;


   diff2=timetaken();

   printf("\nTime taken for memory transfer from host to device in seconds is %f",diff2);

   dim3 dimGrid((VERTICES+BLOCKSIZE-1)/BLOCKSIZE,VERTICES);   
   
   printf("\nPerforming GPU computation");
   

   gettimeofday(&initial,NULL);
   
    for(i=0;i<VERTICES;i++){

    
       CUDA_Kernel<<<dimGrid,BLOCKSIZE>>>(i,cuda_Graph,cuda_Path);
       gpuErrchk( cudaGetLastError()) ;
       cudaThreadSynchronize();
    }


    gettimeofday(&final,NULL);
    
    double diff1=0;
    

    diff1=timetaken();

    printf("\nTime taken for GPU kernel execution in seconds is %f\n",diff1);

    gettimeofday(&initial,NULL);
    
    cudaMemcpy(Device_Graph,cuda_Graph, Graph_Size, cudaMemcpyDeviceToHost);
    cudaMemcpy(Device_Path,cuda_Path, Graph_Size, cudaMemcpyDeviceToHost);
    

    gettimeofday(&final,NULL);
    
    double diff3=0;
    
    diff3=timetaken();
    
    printf("\nTime taken for memory transfer from device to host  in seconds is %f\n",diff3);
    printf("\nTime taken for total memory transfer in seconds is %f\n",(diff2+diff3));

   int match=0;
 
   for(i=0;i<VERTICES*VERTICES;i++){

       if(Host_Graph[i]==Device_Graph[i]){
           match++;

       }

   }
   if(match==(VERTICES*VERTICES)){
       printf("\nThe CPU and GPU results match\n");
   }


   free(Graph);
   free(Host_Graph);
   free(Device_Graph);
   free(Host_Path);
   free(Device_Path);


   cudaFree(cuda_Graph);
   cudaFree(cuda_Path);

}

void Initialize_Graph(int* Graph,int Value){
    uint32_t i,j;
    for(i=0;i<VERTICES;i++){
        for(j=0;j<VERTICES;j++){
           // printf("here %d ",i*VERTICES+j);
            Graph[i*VERTICES + j] = Value;
        }
    }
}

void Arr(int* Input_Array,int Value){
    int i;
    for(i=0;i<VERTICES;i++){
       // printf("ARRAY: %d\n",i);
        Input_Array[i]=Value;
    }
   // printf("over");
}

void Random(int* Graph, int* Edges_Per_Vertex){
    uint32_t i,Current_Edges,Random_Vertex;
    int Random_Dist;

    for(i=1;i<VERTICES;i++){
        Random_Vertex = (rand() % i);
        Random_Dist =(rand() % MAX_DIST) + 1;
        Graph[Random_Vertex*VERTICES + i] = Random_Dist;
        Graph[Random_Vertex + i*VERTICES] = Random_Dist;
        Edges_Per_Vertex[i] += 1;
        Edges_Per_Vertex[Random_Vertex] += 1;
    }

    for(i=0;i<VERTICES;i++){
        Current_Edges = Edges_Per_Vertex[i];
        while(Current_Edges < MIN_EDGES_VERTEX){
            Random_Vertex = (rand() % VERTICES);
            Random_Dist = (rand() % MAX_DIST) + 1;
            if((Random_Vertex != i)&&(Graph[Random_Vertex + i*VERTICES] == 0)){
                Graph[Random_Vertex + i*VERTICES] = Random_Dist;
                Graph[Random_Vertex*VERTICES + i] = Random_Dist;
                Edges_Per_Vertex[i] += 1;
                Current_Edges += 1;
            }
        }
    }

}


