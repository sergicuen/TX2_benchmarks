/////////////////////////////////////////////////////////////////////////////  
// 
//	file: mmult.cu
//  versi�n para radiar en CNA
//	usage: mat_mul_redundant -s SIZE -x threads_per_block_x -y threads_per_block_y -b replicate_input -r RunsBlock
//
/////////////////////////////////////////////////////////////////////////////  
//	Descripci�n: Genera 2 kernels concurrentes que realizan C=AxB
//  - if (duplicate_input==1)
//			kernel_1: C = A x B
//			kernel_2: C'= A' x B'
//			Check_DMR: check(C==C')
//  - else
//			kernel_1: C = A x B
//			kernel_2: C'= A x B
//			Check_DMR: check(C==C')
//
//  M�dulos a�adidos:
//  - VarInit: inicializa las matrices con valores conocidos para que el resultado 
//		de AxB pueda ser verificado por la CPU de forma r�pida
//  - check_golden: genera los resultados C_golden con la CPU y comprueba si 
//		coinciden con los resultados GPU
//  - c�lculo del CRC de los resultados (todo)
//  
//	Funcionalidad adicional:
//  - A�adida verificaci�n adicional:
//		if (check_DMR=correct) then 
//			if (check_golden ==correct) then Error_Masked
//			else SDC_undetected
//		else SDC_detected
//
////////////////////////////////////////////////////////////////////////////////////////////
//
//  TODO: 
//	- verificar que lo c�lculos se realizan aunque los datos de entrada no hayan cambiado
//  - bajar los clocks
//  - SOLO FUNCIONA CON NUM_STREAMS 2


#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include<sys/time.h>

#define USE_GOLDEN 0    // inicializa mat para comprobaci�n con golden en CPU
#define PERFORMANCE 0   // muestra tiempos de ejecuci�n
#define ROBUST_PRINTING 0 // muestra detalles de errores
#define INJECT_FAULTS 0
//#define RUNBLOCK 1000  // par�metro de entrada

using namespace std;

#ifdef REDUNDANT
#define NUM_STREAMS 2
#else
#define NUM_STREAMS 1
#endif

///TIMING VARIABLES
struct timeval time_start;
struct timeval time_end;
struct timeval Kernel1_start;
struct timeval Kernel1_end;
//DEVICE NUMBER OF ERRORS VARIABLE
__device__ int d_num_errors = 0;



///COMPUTE TIMING FUNCTION
long int get_time(struct timeval time_start, struct timeval time_end){
	long int r ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	return r;
}


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

__global__ void compute_difference(void *C1, void *C2, int N){
  float epsilon = 0.001;
  //Calculate ROW and COLUMN
  int ROW = (int) (blockIdx.y * blockDim.y + threadIdx.y);
  int COL = (int) (blockIdx.x * blockDim.x + threadIdx.x);

  float * ptr_C1 = ((float *)C1);
  float * ptr_C2 = ((float *)C2);
  //Check inside the matrix
  if (ROW < N && COL < N) {
    if (std::abs((ptr_C1[ROW * N + COL]) - (ptr_C2[ROW * N + COL])) < epsilon){
      atomicAdd(&d_num_errors, 1);
    }
  }
}

__global__ void matrixMultiplicationKernel(void *A, void *B, void *C, int N, int ITERACIONES_KERNEL){
    
    int ROW = (int) (blockIdx.y * blockDim.y + threadIdx.y);
    int COL = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    
    float * ptr_A = ((float *)A);
    float * ptr_B = ((float *)B);
    float * ptr_C = ((float *)C);
    for (int i=0; i < ITERACIONES_KERNEL; ++i){
      float tmpSum = 0;

      if (ROW < N && COL < N) {
        if (i == 0)ptr_C[ROW * N + COL] = 0;
        for (int i = 0; i < N; ++i) {
            tmpSum += ptr_A[ROW * N + i] * ptr_B[i * N + COL];
        }
        ptr_C[ROW * N + COL] = tmpSum;
      }
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

//Inicializaci�n de matrices para check_error en CPU
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


void check_and_parse(int argc, char* argv[] , int * N , int * threads_per_blockx , int * threads_per_blocky, int * rblock, bool * replicate_input, int *ITERACIONES_KERNEL, bool *COMPARACION_GPU){

  if (argc !=13  && argc !=15){
    printf("usage: mat_mul_redundant -s SIZE -x threads_per_block_x -y threads_per_block_y -b replicate_input -k ITERACIONES_KERNEL -g COMPARACION_GPU -r RunsBlock\n");
    exit(EXIT_FAILURE);
  }

  int opt;

  while ((opt = getopt(argc, argv, "s:x:y:r:b:k:g:")) != -1){
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
      case 'k':
        if (optarg) *ITERACIONES_KERNEL = atoi(optarg);
      break;
      case 'g':
        if(optarg) *COMPARACION_GPU = (atoi(optarg) == 1);
      break;
      default: /* '?' */
        
        exit(EXIT_FAILURE);
    }
  }
}


void mem_alloc_host(void ** h_A, void ** h_B , void ** h_C1, void ** h_C2,  int  SIZE){
  
  HANDLE_ERROR( cudaHostAlloc( h_A, SIZE, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaHostAlloc( h_B, SIZE, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaHostAlloc( h_C1, SIZE, cudaHostAllocDefault ) );
  HANDLE_ERROR( cudaHostAlloc( h_C2, SIZE, cudaHostAllocDefault ) );
}

void mem_alloc_device(void ** d_A1, void ** d_B1,void ** d_C1, void ** d_C2,void ** d_A2, void ** d_B2,  bool replicate_input , int  SIZE , void ** d_pA, void ** d_pB){
  
  HANDLE_ERROR( cudaMalloc( d_A1, SIZE ) );
  HANDLE_ERROR( cudaMalloc( d_B1, SIZE ) );
  HANDLE_ERROR( cudaMalloc( d_C1, SIZE ) );
  HANDLE_ERROR( cudaMalloc( d_C2, SIZE ) );
  if (replicate_input){
    HANDLE_ERROR( cudaMalloc( d_A2, SIZE ) );
    HANDLE_ERROR( cudaMalloc( d_B2, SIZE ) );
  }
  d_pA[0] = d_A1[0];
  d_pB[0] = d_B1[0] ; 
  if (replicate_input){
    d_pA[0] = d_A2[0] ; 
    d_pB[0] = d_B2[0] ;
  }

}

void mem_free_host(void *h_A, void *h_B, void *h_C1, void *h_C2){
  HANDLE_ERROR( cudaFreeHost( h_A));
  HANDLE_ERROR( cudaFreeHost( h_B));
  HANDLE_ERROR( cudaFreeHost( h_C1));
  HANDLE_ERROR( cudaFreeHost( h_C2));
}

void mem_free_device(void * d_A1,void * d_B1,void *d_A2,void *d_B2 ,void * d_C1 ,void * d_C2 , bool replicate_input){
 
  HANDLE_ERROR( cudaFree( d_A1 ) );
  HANDLE_ERROR( cudaFree( d_B1 ) );

  if (replicate_input){
  HANDLE_ERROR( cudaFree( d_A2 ) );
  HANDLE_ERROR( cudaFree( d_B2 ) );
  }

  HANDLE_ERROR( cudaFree( d_C1 ) );
  HANDLE_ERROR( cudaFree( d_C2 ) );
}


void mem_copy_to_device(void * d_A1 , void * d_A2, void * d_B1 , void * d_B2 ,  void * h_A , void * h_B, size_t SIZE, bool replicate_input , cudaStream_t *  stream  ){
  
  HANDLE_ERROR(cudaMemcpyAsync(d_A1,h_A,SIZE,cudaMemcpyHostToDevice,stream[0]));
  HANDLE_ERROR(cudaMemcpyAsync(d_B1,h_B,SIZE,cudaMemcpyHostToDevice,stream[0]));
#ifdef REDUNDANT
  if (replicate_input){
    HANDLE_ERROR( cudaMemcpyAsync( d_A2, h_A,SIZE, cudaMemcpyHostToDevice, stream[1]) );
    HANDLE_ERROR( cudaMemcpyAsync( d_B2, h_B,SIZE, cudaMemcpyHostToDevice, stream[1]) );
    }
#endif
}


void mem_copy_to_host(void * h_C1,void * h_C2,  void *  d_C1, void * d_C2,  size_t SIZE ,cudaStream_t *  stream ){
    
  HANDLE_ERROR( cudaMemcpyAsync( h_C1, d_C1,SIZE, cudaMemcpyDeviceToHost, stream[0] ));
#ifdef REDUNDANT
 HANDLE_ERROR( cudaMemcpyAsync( h_C2, d_C2,SIZE, cudaMemcpyDeviceToHost, stream[1] ));
#endif

}

//Revisa toda la matriz y env�a n�mero de errores detectados
bool test( void * h_C1, void * h_C2, int N){
  bool correct = true ;
  int   local_errors=0;
  for (int i = 0; i < N ; ++i){
    for (int j = 0; j < N; ++j){
      correct = (((float*)h_C1)[j + i*N] == ((float*)h_C2)[j + i*N]);
	  if (correct == false){
          local_errors++;
          #if ROBUST_PRINTING
            printf("ERROR detected at C[%d][%d]\n", i, j);
          #endif
      }
    }
  }
  if (local_errors !=0) {
      printf("- #run: %u\n", runs_counter);
	  printf("ERRORS detected: %d\n", local_errors);
      correct=false;
  }

  return correct;
}

// Revisa  la matriz y salta al primer error
bool FindFirstError( void * h_C1, void * h_C2, int N){
  bool correct = true ;
  
  for (int i = 0; i < N && correct==true; ++i){
    for (int j = 0; j < N && correct==true; ++j){
      correct = (((float*)h_C1)[j + i*N] == ((float*)h_C2)[j + i*N]);
	  if (correct == false){
          printf("- #run: %u\n", runs_counter);
          printf("ERROR detected at C[%d][%d]\n", i, j);
      }
    }
  }

  return correct;
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
      printf("- #run: %u\n", runs_counter); 
      printf(" ERRORS: %u\n", local_errors);  
      correct=false;
    }
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
  gettimeofday(&time_start, NULL);

  /*
  * Data
  */
  int N;
  int threads_per_blockx;
  int threads_per_blocky;
#ifdef REDUNDANT 
  int num_errors = 0;
#endif
  struct timeval time_start;
  struct timeval time_end;
  struct timeval time_compare1;
  struct timeval time_compare2;
  struct timeval Transfer1_start;
  struct timeval Transfer1_end;
  struct timeval Transfer2_start;
  struct timeval Transfer2_end;

  unsigned int TotalexTime;
  //By putting it to false we avoid redundant memory copies in not-redundant version
  bool replicate_input = false;
#ifdef REDUNDANT
  replicate_input = true;
#endif
  int rblock=1;
  bool correct_C1=true;
  bool correct_C2=true;
  bool DMR_correct=true;
  bool COMPARACION_GPU = 0;
  int ITERACIONES_KERNEL = 1;
  /*
  * Checking arguments used to call the program
  */
  check_and_parse(argc,argv,&N,&threads_per_blockx,&threads_per_blocky,&rblock,&replicate_input, &ITERACIONES_KERNEL, &COMPARACION_GPU);
  #if ROBUST_PRINTING
    printf("Hw: Jetson nano, Maxwell arch \r\n");
    printf("Test: MMULT_DMR\r\n");
    if (replicate_input) printf("Replicating the input for each kernel\n");
    printf("Version: 1.0 \r\n");
    printf("matrix size:%d\n", N);  
    printf(" data type: float\r\n");
  #else
    printf("Hw:Jn, T:MMULT_DMR, V:%d, ThBlck:%d, Sz:%d, Dt:fp, RunsB:%d\r\n", replicate_input, threads_per_blockx, N, rblock);
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
  void *h_A, *h_B, *h_C1, *h_C2;
  mem_alloc_host(&h_A,&h_B,&h_C1,&h_C2, SIZE * sizeof(float));

  /*
  * Initialize matrices on the host
  */
#if USE_GOLDEN
  initMat4Golden(h_A, N, N);
  initMat4Golden(h_B, N, N);
#else
  if (NUM_STREAMS==1) {initMat4Golden(h_A, N, N); initMat4Golden(h_B, N, N);}
  else initMatrices(h_A,h_B,N);
#endif

//while (1){
for (runs_counter=0; runs_counter < rblock; runs_counter++){


    //printf("MaT_A\n");
    //printMatrix( h_A,  N);

  /*
  * Allocate memory on the device. 
  * Create pointers to data structure
  * Create two temporary pointers for replicate input purpose
  */
  void *d_A1, *d_B1, *d_A2, *d_B2, *d_C1, *d_C2, *d_pA , *d_pB;
  mem_alloc_device(&d_A1,&d_B1,&d_C1,&d_C2,&d_A2,&d_B2,replicate_input, SIZE * sizeof(float),&d_pA,&d_pB);
  
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
  cudaStream_t stream[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; ++i)
    HANDLE_ERROR( cudaStreamCreate(&stream[i]));

  /*
  * Copy the arrays 'a' and 'b' to the GPU
  */
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&Transfer1_start, NULL);
  mem_copy_to_device(d_A1,d_A2,d_B1,d_B2,h_A,h_B,SIZE * sizeof(float),replicate_input,&stream[0]);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&Transfer1_end, NULL);

  /*
  * Sync streams 
  */
  HANDLE_ERROR (cudaStreamSynchronize(stream[0]) );

#ifdef REDUNDANT
   HANDLE_ERROR (cudaStreamSynchronize(stream[1]) );
#endif

// gettimeofday(&time_start, NULL);

 HANDLE_ERROR( cudaDeviceSynchronize() );
  //Get timing
  gettimeofday(&Kernel1_start, NULL);
  /*
  * Kernels launch, second kernel execution pointers depends on replication_input value
  */
  
  matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock, 0, stream[0]>>>(d_A1, d_B1, d_C1, N, ITERACIONES_KERNEL);
  HANDLE_ERROR( cudaPeekAtLastError() );

#ifdef SERIALIZE
  HANDLE_ERROR( cudaDeviceSynchronize() );
#endif

#ifdef REDUNDANT
  matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock, 0, stream[1]>>>(d_pA, d_pB, d_C2, N, ITERACIONES_KERNEL);
  // SC? esto debería ir fuera del ifdef??
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );
#endif
HANDLE_ERROR( cudaDeviceSynchronize() );
gettimeofday(&Kernel1_end, NULL);


// SC? hay que hacer las dos cosas cudaDeviceSynchronize() y cudaStreamSynchronize(stream[0])??


///////// SC Check the results   //////////////////////////////////////////////////////////////
bool correct = true;
if (!COMPARACION_GPU){
  /*
  * Sync streams 
  */
  HANDLE_ERROR (cudaStreamSynchronize(stream[0]) );
  
  #ifdef REDUNDANT
   HANDLE_ERROR (cudaStreamSynchronize(stream[1]) );
  #endif

  /*
  * Copy back results data from device to host memory
  */
  gettimeofday(&Transfer2_start, NULL);
  mem_copy_to_host(h_C1,h_C2,d_C1,d_C2, SIZE*sizeof(float),&stream[0]);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&Transfer2_end, NULL);

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
#if USE_GOLDEN
  correct_C1= Golden_check(h_C1, N, N, N);
  correct = correct_C1;
#ifdef REDUNDANT
  correct_C2= Golden_check(h_C2, N, N, N);
  correct &= correct_C2;
#endif
#endif

#ifndef REDUNDANT 
    correct_C1= Golden_check(h_C1, N, N, N);
    correct = correct_C1;
#else
    DMR_correct = test(h_C1,h_C2,N);           // the comparison time is long and always the same
    //bool DMR_correct = FindFirstError(h_C1,h_C2,N);   // the comparison time is lower in average
    correct = DMR_correct;
#endif
  gettimeofday(&time_compare2, NULL);
} //END COMPARISON BY CPU

else { //COMPARISON BY GPU

#ifdef REDUNDANT //No sense on perform comparison in GPU for non-redundant version
  gettimeofday(&time_compare1, NULL);
  //Compute differences
  compute_difference<<< blocksPerGrid,threadsPerBlock >>>(d_C1, d_C2, N);
  HANDLE_ERROR( cudaPeekAtLastError() );
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&time_compare2, NULL);

  gettimeofday(&Transfer2_start, NULL);

  //Pass the result to the HOST
  cudaMemcpyFromSymbol(&num_errors, "d_num_errors", sizeof(num_errors), 0, cudaMemcpyDeviceToHost);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&Transfer2_end, NULL);

  //Update correct
  correct = (num_errors == 0);

#endif
}
  /*
  * Free host memory
  */
  // Esto lo hago s�lo al final del bloque
  //mem_free_host(h_A,h_B,h_C1,h_C2);
  
  /*
  * Sync streams 
  */
  for (int i = 0; i < NUM_STREAMS; ++i)
    HANDLE_ERROR(cudaStreamSynchronize(stream[i])); // SC he puesto HANDLE_ERROR

  /*
  * Destroy streams 
  */
  for (int i = 0; i < NUM_STREAMS; ++i)
    HANDLE_ERROR( cudaStreamDestroy(stream[i]));

  //SC meto esta l�nea para provocar error y ver que devuelve
  //HANDLE_ERROR( cudaStreamDestroy(stream[0]));
  /*
  * Free device GPU memory
  */
  mem_free_device(d_A1,d_B1,d_A2,d_B2,d_C1,d_C2,replicate_input);


   gettimeofday(&time_end, NULL);

  
    
  #if PERFORMANCE
    //unsigned int exTime=0;
    //exTime = (unsigned int) ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
    printf("Comparison time: %u us\n", (unsigned int) ((time_compare2.tv_sec * 1000000 + time_compare2.tv_usec) - (time_compare1.tv_sec * 1000000 + time_compare1.tv_usec)));
    //printf("Execution time of both kernels: %u us\n", exTime);
  #endif

//////////////// SC PRINT RESULTS ///////////////////////////////////////////////////
runs_counter++;
#if USE_GOLDEN
  if (correct==true) { 
      if (correct_C1==false || correct_C2==false) {
          printf("ERRORS undetected\n");
          initMat4Golden(h_A, N, N);
          initMat4Golden(h_B, N, N);
          runs_werror++;
      }
  }
  else { printf("ERRORS detected by DMR\n");
  }
#else
  if (NUM_STREAMS == 1) {
    if (correct==true) {
      printf("OK\n");
    }
    else {
      printf("ERRORS\n");
      initMat4Golden(h_A, N, N);
      initMat4Golden(h_B, N, N);
      runs_werror++;
    } 
  }
  else {
	if (correct==true && runs_counter % rblock==0 && runs_counter !=0) {
		TotalexTime = (unsigned int) ((time_end.tv_sec) - (time_start.tv_sec));
		printf("TEST_CHECK:%u;RUNS_WERROR:%d; EXEC_TIME:%us\n", rblock, runs_werror, TotalexTime);
	
    //if (correct==true) {printf("OK\n");
    }
	else {
      printf("ERRORS detected by DMR\n");
      initMatrices(h_A,h_B,N);
      runs_werror++;
    }
  }
#endif

gettimeofday(&time_end, NULL);

long int TotalExecutionTime = get_time(time_start, time_end);
printf("Execution time: %ld us\n", TotalExecutionTime);
long int Kernel1;
Kernel1 = get_time(Kernel1_start, Kernel1_end);
long int TotalKernelExecutionTime = Kernel1;
#ifdef REDUNDANT
if (COMPARACION_GPU){
  //In this case the comparison is done in the GPU
  TotalKernelExecutionTime += get_time(time_compare1, time_compare2);
}
#endif
long int TotalTransferTime = get_time(Transfer1_start, Transfer1_end);
TotalTransferTime += get_time(Transfer2_start, Transfer2_end);
printf("Time for CUDA kernels:\t%ld us\n",TotalKernelExecutionTime);
printf("Transfer Time: %ld, Kernel Time: %ld, Execution Time: %ld\n", TotalTransferTime, TotalKernelExecutionTime, TotalExecutionTime);
printf("GPU time (Transfers + Kernels): %f%%\n", (TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100);
printf("CPU time (rest): %f%%\n", 100 - ((TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100));

}
mem_free_host(h_A,h_B,h_C1,h_C2);
// gettimeofday(&time_end, NULL);
// TotalexTime = (unsigned int) ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
// TotalexTime = (unsigned int) ((time_end.tv_sec) - (time_start.tv_sec));
//printf("TEST_CHECK:%u;RUNS_WERROR:%d; EXEC_TIME:%us\n", rblock, runs_werror, TotalexTime);


  return 0;
}
