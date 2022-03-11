/////////////////////////////////////////////////////////////////////////////  
// 
//	file: mmult_sec.cu
//  versión para radiar en CNA
//	usage: mat_sec -s SIZE -x threads_per_block_x -y threads_per_block_y -b replicate_input -r RunsBlock
//
/////////////////////////////////////////////////////////////////////////////  
//	Descripción: Genera 2 kernels concurrentes que realizan C=AxB
//  - if (duplicate_input==1)
//			kernel_1: C = A x B
//			kernel_2: C'= A' x B'
//			Check_DMR: check(C==C')
//  - else
//			kernel_1: C = A x B
//			kernel_2: C'= A x B
//			Check_DMR: check(C==C')
//
//  Módulos añadidos:
//  
//	Funcionalidad adicional:
//
////////////////////////////////////////////////////////////////////////////////////////////
//
//  TODO: 
//  - bajar los clocks


#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include<sys/time.h>

#define NUMBER_OF_STREAMS 1
#define USE_GOLDEN 0    // inicializa mat para comprobación con golden en CPU
#define PERFORMANCE 0   // muestra tiempos de ejecución
#define ROBUST_PRINTING 0 // muestra detalles de errores
#define INJECT_FAULTS 0
//#define RUNBLOCK 1000  // parámetro de entrada

using namespace std;

unsigned int runs_counter=0;
unsigned int runs_werror=0;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "-run: %d; %s in %s at line %d\n", runs_counter, cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


__global__ void matrixMultiplicationKernel(void *A, void *B, void *C, int N){
    
    int ROW = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int COL = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    
    float * ptr_A = ((float *)A);
    float * ptr_B = ((float *)B);
    float * ptr_C = ((float *)C);

    float tmpSum = 0;

    if (ROW < N && COL < N) {
//#pragma unroll 1
        for (int i = 0; i < N; ++i) {
            tmpSum += ptr_A[ROW * N + i] * ptr_B[i * N + COL];
        }
        ptr_C[ROW * N + COL] = tmpSum;
    }
}

void initMatrices( void *h_A, void *h_B, int N ){
  // Initialize matrices on the host
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
    ((float*)h_A)[i*N+j] = i;
    ((float*)h_B)[i*N+j] = N-i;
    }
  }
}

//Inicialización de matrices para check_error en CPU
void initMat4Golden(void *data, int n, int m) {
    int ct = 0;
    float val, tmp;
    for (int i = 1; i < n+1; ++i) {
      val = i;
      for (int j = 1; j < m+1; ++j) {
         tmp= val/(n*m);
		 ((float*)data)[ct++]=tmp;
        val += i;
      }
    }
}

void inject_faults(void *h_C, int N){
  ((float*)h_C)[2*N+1] = 7.5;
  ((float*)h_C)[N] = 7.5;
}


void check_and_parse(int argc, char* argv[] , int * N , int * threads_per_blockx , int * threads_per_blocky, int * rblock, bool * replicate_input){

  if (argc !=9  && argc !=11){
    printf("usage: mat_mul_redundant -s SIZE -x threads_per_block_x -y threads_per_block_y -b replicate_input -r RunsBlock\n");
    exit(EXIT_FAILURE);
  }

  int opt;

  while ((opt = getopt(argc, argv, "s:x:y:r:b")) != -1){
    switch (opt) {
      case 's':
        if(optarg) *N = (unsigned long long) atoi(optarg);
        break;
      case 'x': 
        if(optarg) *threads_per_blockx = atoi(optarg);
        break;
      case 'y': 
        if(optarg) *threads_per_blocky = atoi(optarg);
        break;
      case 'r':
        if(optarg) *rblock = (unsigned int) atoi(optarg);
        break;
      case 'b':
        if(optarg) *replicate_input = (atoi(optarg) == 1);
        break;

      default: /* '?' */
        
        exit(EXIT_FAILURE);
    }
  }
}

// void mem_alloc_host_op(void ** h_A, void ** h_B, int  SIZE){
  
  // HANDLE_ERROR( cudaHostAlloc( h_A, SIZE, cudaHostAllocDefault ) );
  // HANDLE_ERROR( cudaHostAlloc( h_B, SIZE, cudaHostAllocDefault ) );
// }

// void mem_alloc_host_res(void ** h_C1, int  SIZE){
  // HANDLE_ERROR( cudaHostAlloc( h_C1, SIZE, cudaHostAllocDefault ) );
// }

void mem_alloc_host(void ** h_A, void ** h_B , void ** h_C1, int  SIZE){
  
  HANDLE_ERROR( cudaHostAlloc( h_A, SIZE, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaHostAlloc( h_B, SIZE, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaHostAlloc( h_C1, SIZE, cudaHostAllocDefault ) );

}



void mem_alloc_device(void ** d_A1, void ** d_B1,void ** d_C1, int  SIZE , void ** d_pA, void ** d_pB){
  
  HANDLE_ERROR( cudaMalloc( d_A1, SIZE ) );
  HANDLE_ERROR( cudaMalloc( d_B1, SIZE ) );
  HANDLE_ERROR( cudaMalloc( d_C1, SIZE ) );

  d_pA[0] = d_A1[0];
  d_pB[0] = d_B1[0] ; 

}

void mem_free_host(void *h_A, void *h_B, void *h_C1){
  HANDLE_ERROR( cudaFreeHost( h_A));
  HANDLE_ERROR( cudaFreeHost( h_B));
  HANDLE_ERROR( cudaFreeHost( h_C1));
}

// void mem_free_host_op(void *h_A, void *h_B){
  // HANDLE_ERROR( cudaFreeHost( h_A));
  // HANDLE_ERROR( cudaFreeHost( h_B));
// }

// void mem_free_host_res(void *h_C1){
  // HANDLE_ERROR( cudaFreeHost( h_C1));
// }


void mem_free_device(void * d_A1,void * d_B1,void * d_C1 ){
 
  HANDLE_ERROR( cudaFree( d_A1 ) );
  HANDLE_ERROR( cudaFree( d_B1 ) );
  HANDLE_ERROR( cudaFree( d_C1 ) );
}


void mem_copy_to_device(void * d_A1 ,  void * d_B1 , void * h_A , void * h_B, size_t SIZE, cudaStream_t *  stream  ){
  
  HANDLE_ERROR(cudaMemcpyAsync(d_A1,h_A,SIZE,cudaMemcpyHostToDevice,stream[0]));
  HANDLE_ERROR(cudaMemcpyAsync(d_B1,h_B,SIZE,cudaMemcpyHostToDevice,stream[0]));

}


void mem_copy_to_host(void * h_C1, void *  d_C1, size_t SIZE ,cudaStream_t *  stream ){
    
  HANDLE_ERROR( cudaMemcpyAsync( h_C1, d_C1,SIZE, cudaMemcpyDeviceToHost, stream[0] ));

}



// check_error en CPU
bool Golden_check(void *h_C, int m, int n, int k) {
    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero
//    double eps = 1.e-10 ; // machine zero
    bool correct = true;
    unsigned int local_errors=0;
//    const float factor = ( 1.0f*n * 1.0f*(n+1) * 1.0f*(2*n+1) )/ (6.0f*k*k);
    const float factor = ( 1.0f*n * 1.0f*(n+1) * 1.0f*(2*n+1) )/ (6.0f*k*k*k*k);
    int ct = 0;
    for (int i = 1; i < m+1; i++) {
      for (int j = 1; j < n+1; j++) {
        double abs_err = fabs(((float*)h_C)[ct] - (i*j*factor));
        double dot_length = k;
        double abs_val = fabs(((float*)h_C)[ct]);
        double rel_err = abs_err/abs_val/dot_length;
		
        if (rel_err > eps)
        {
            // printf("- #run: %u\n", runs_counter);            
            // printf("ERROR: Matrix[%d][%d]=%.8f, ref=%.8f error term is > %E\n", 
               // i-1, j-1, ((float*)h_C)[ct], i*j*factor, eps);
            // correct = false;
          local_errors++;

        }
        ct++;
      }
    }
    if (local_errors !=0){
      printf("\n- #run: %u\n", runs_counter); 
      printf(" ERRORS: %u\n", local_errors);  
      correct=false;
    }
    //else {printf("C");}
    return correct;
}


void printMatrix( void * matrix , int N){
  for(int i = 0 ; i < N ; i++){
    for(int j=0; j<N ; j++){
      printf("%i ", (int)((float *)matrix)[i * N + j ]);
	  //printf("%.8f ", ((float *)matrix)[i * N + j ]);
    }
    printf("\n");
  }
}

int main (int argc, char* argv[]) {

  /*
  * Data
  */
  int N;
  int threads_per_blockx;
  int threads_per_blocky;
  struct timeval time_start;
  struct timeval time_end;
  struct timeval time_compare1;
  struct timeval time_compare2;
  unsigned int TotalexTime;

  bool replicate_input = true;
  int rblock=10;
  bool correct_C1=true;
  /*
  * Checking arguments used to call the program
  */
  check_and_parse(argc,argv,&N,&threads_per_blockx,&threads_per_blocky,&rblock,&replicate_input);
  #if ROBUST_PRINTING
    printf("Hw: Jetson nano, Maxwell arch \r\n");
    printf("Test: MMULT_SEC\r\n");
    if (replicate_input) printf("Replicating the input for each kernel\n");
    printf("Version: 1.0 \r\n");
    printf("matrix size:%d\n", N);  
    printf(" data type: float\r\n");
  #else
    printf("Hw:Jn, T:MMULT_SEC_, V:%d, ThBlck:%d, Sz:%d, Dt:fp, RunsB:%d\r\n", replicate_input, threads_per_blockx, N, rblock);
  #endif
  
  /*
  * Perform matrix multiplication C = A*B
  * where A, B and C are NxN matrices
  * The matrices are floating point values
  */ 
  int SIZE = (int)N*N;
  if(SIZE < 0){ printf("Size Overflow\n"); exit(EXIT_FAILURE);}
  //printf("Matrix of size: %d\n",SIZE);

gettimeofday(&time_start, NULL);

  /*
  * Allocate pinned memory on the CPU to make asynchronous transfers.
  */
  void *h_A, *h_B, *h_C1;
  void *d_A1, *d_B1, *d_C1, *d_pA , *d_pB;
  mem_alloc_host(&h_A,&h_B,&h_C1, SIZE * sizeof(float));
  // mem_alloc_host_op(&h_A,&h_B, SIZE * sizeof(float));

  /*
  * Initialize matrices on the host
  */
  initMat4Golden(h_A, N, N);
  initMat4Golden(h_B, N, N);

for (runs_counter=0; runs_counter < rblock; runs_counter++){


    //printf("MaT_A\n");
    //printMatrix( h_A,  N);

  /*
  * Allocate memory on the device. 
  * Create pointers to data structure
  * Create two temporary pointers for replicate input purpose
  */
  //void *d_A1, *d_B1, *d_C1, *d_pA , *d_pB;
  // mem_alloc_host_res(&h_C1, SIZE * sizeof(float));
  mem_alloc_device(&d_A1,&d_B1,&d_C1, SIZE * sizeof(float),&d_pA,&d_pB);
  
  /*
  * Setting the block, grid dimension for GPU computation
  */
  int blocks_count_x = (N + threads_per_blockx - 1)/ threads_per_blockx;
  int blocks_count_y = (N + threads_per_blocky - 1)/ threads_per_blocky;  
  dim3 threadsPerBlock(threads_per_blockx, threads_per_blocky);
  dim3 blocksPerGrid(blocks_count_x, blocks_count_y);

  /*
  * Creating the streams for the kernel launch
  */
  cudaStream_t stream[NUMBER_OF_STREAMS];
  for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
    HANDLE_ERROR( cudaStreamCreate(&stream[i]));

  /*
  * Copy the arrays 'a' and 'b' to the GPU
  */
  mem_copy_to_device(d_A1,d_B1,h_A,h_B,SIZE * sizeof(float),&stream[0]);

  /*
  * Sync streams 
  */
  HANDLE_ERROR (cudaStreamSynchronize(stream[0]) );
 // gettimeofday(&time_start, NULL);

  /*
  * Kernels launch, second kernel execution pointers depends on replication_input value
  */
  matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock, 0, stream[0]>>>(d_A1, d_B1, d_C1, N);

  /*
  * Sync streams 
  */
  HANDLE_ERROR (cudaStreamSynchronize(stream[0]) );

  // gettimeofday(&time_end, NULL);

  /*
  * Copy back results data from device to host memory
  */
  mem_copy_to_host(h_C1,d_C1, SIZE*sizeof(float),&stream[0]);
  

  /*
  * Sync streams 
  */
  for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
    HANDLE_ERROR(cudaStreamSynchronize(stream[i])); // SC he puesto HANDLE_ERROR

  /*
  * Destroy streams 
  */
  for (int i = 0; i < NUMBER_OF_STREAMS; ++i)
    HANDLE_ERROR( cudaStreamDestroy(stream[i]));

  //SC meto esta línea para provocar error y ver que devuelve
  //HANDLE_ERROR( cudaStreamDestroy(stream[0]));
  /*
  * Free device GPU memory
  */
  mem_free_device(d_A1,d_B1,d_C1);
  //mem_free_host_res(h_C1); //SC*********************


#if INJECT_FAULTS
  if (runs_counter==2 || runs_counter==40){
    inject_faults(h_C1,N);
  }
#endif
  /*
  *Start timing
  */
  gettimeofday(&time_compare1, NULL);
  // printf("MaT_C\n");
  // printMatrix( h_C1,  N);

  /*
  * Checking if the two output arrays are the same
  */
  correct_C1= Golden_check(h_C1, N, N, N);

  gettimeofday(&time_compare2, NULL);
  
  /*
  * Free host memory
  */
  // Esto lo hago sólo al final del bloque
  //mem_free_host(h_A,h_B,h_C1,h_C2);
  

    gettimeofday(&time_end, NULL);
    
  #if PERFORMANCE
    //unsigned int exTime=0;
    //exTime = (unsigned int) ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    printf("Comparison time: %u us\n", (unsigned int) ((time_compare2.tv_sec * 1000000 + time_compare2.tv_usec) - (time_compare1.tv_sec * 1000000 + time_compare1.tv_usec)));
    //printf("Execution time of both kernels: %u us\n", exTime);
  #endif

  if (correct_C1==true) {
      printf("C\r\n");
  }
  else {
      //printf("\nERRORS\n");
      initMat4Golden(h_A, N, N);
      initMat4Golden(h_B, N, N);
      runs_werror++;
  } 

}

mem_free_host(h_A,h_B,h_C1);
// mem_free_host_op(h_A,h_B);

gettimeofday(&time_end, NULL);
//TotalexTime = (unsigned int) ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
TotalexTime = (unsigned int) ((time_end.tv_sec) - (time_start.tv_sec));
printf("\r\nTEST_CHECK:%u;RUNS_WERROR:%d; EXEC_TIME:%us\n", rblock, runs_werror, TotalexTime);


  return 0;
}
