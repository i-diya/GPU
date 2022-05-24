#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include<thread>


#define VERTICES (10)           
#define MIN_EDGES_VERTEX (2)    
#define INF_DIST (10000000)     
#define MAX_DIST (1000)         
int BLOCKSIZE=512;              


void Random(int* Graph, int* Edges_Per_Vertex);


void dis(int* Graph,int Value);


void arr(int* Input_Array,int Value);

void Serial_Floyd(int* Graph1,int *Graph_Path, int size);

double timetaken();


struct timeval initial, final;

 void Serial_Floyd(int* Graph1,int *Graph_Path, int size){
     int x,y,z;

     for(x=0;x<VERTICES;++x){
       for(y=0;y<VERTICES;++y){
             for(z=0;z<VERTICES;++z){
 		        int current_node=y*VERTICES+z;
		        int Node_i=y*VERTICES+x;
		        int Node_j=x*VERTICES+z;
                if(Graph1[current_node]>(Graph1[Node_i]+Graph1[Node_j])){
                    Graph1[current_node]=(Graph1[Node_i]+Graph1[Node_j]);
                    Graph_Path[current_node]=x;
                }
            }
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
   srand(8421);

   /*Host memory allocation*/
   int Graph_Size=VERTICES*VERTICES*sizeof(int);
   int *Graph=(int *)malloc(Graph_Size);
   int *Host_Graph=(int *)malloc(Graph_Size);
   int *Host_Path=(int *)malloc(Graph_Size);
   int *Device_Graph=(int *)malloc(Graph_Size);
   int *Device_Path=(int *)malloc(Graph_Size);
   int* Edges_Per_Vertex = (int*)malloc(VERTICES*sizeof(int));
   printf("\nBlocksize :%d",BLOCKSIZE);
   printf("\nVertices  :%d",VERTICES);

   
   dis(Graph,3);
   arr(Edges_Per_Vertex,3);

   Random(Graph,Edges_Per_Vertex);


   free(Edges_Per_Vertex);


   int i;
   for(i=0;i<VERTICES*VERTICES;i++){
       Host_Graph[i]=Graph[i];
       Host_Path[i]=-1;
   }
   printf("\nPerforming CPU computation");

   gettimeofday(&initial,NULL);

   int x,y,z;
   #pragma omp target enter data map(to: Host_Graph[0:VERTICES*VERTICES]) map(tofrom: Host_Path[0:VERTICES*VERTICES]) 
   #pragma omp target teams distribute parallel for simd
   for(x=0;x<VERTICES;++x){
       for(y=0;y<VERTICES;++y){
           for(z=0;z<VERTICES;++z){
		       int current_node=y*VERTICES+z;
		       int Node_i=y*VERTICES+x;
		       int Node_j=x*VERTICES+z;
               if(Host_Graph[current_node]>(Host_Graph[Node_i]+Host_Graph[Node_j])){
                   Host_Graph[current_node]=(Host_Graph[Node_i]+Host_Graph[Node_j]);
                   Host_Path[current_node]=x;
               }
           }
       }
   }
  // Serial_Floyd(Graph,Host_Path,0);

  /*   for(i=0;i<VERTICES*VERTICES;i++){
      if( Host_Graph[i]!=Graph[i]){printf("NOT EQUAL GRAPHS\n");}
       
   }*/

   gettimeofday(&final,NULL);
   double diff=0;

   diff=timetaken();

   printf("\nTime taken for logic computation by CPU in seconds is %f",diff);

}
    

void dis(int* Graph,int Value){
    uint32_t i,j;
    for(i=0;i<VERTICES;i++){
        for(j=0;j<VERTICES;j++){
            Graph[i*VERTICES + j] = Value;
        }
    }
   // printf("here\n");
}


void arr(int* Input_Array,int Value){
    uint32_t i;
    for(i=0;i<VERTICES;i++){
        Input_Array[i]=Value;
    }
   //   printf("here\n");
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


