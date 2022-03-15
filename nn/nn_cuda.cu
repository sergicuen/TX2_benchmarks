/****************************************************************************************
 * nn.cu
 * Nearest Neighbor
 * Auth: SA, SC
 * Modified for radiation tests 15/03/2022
 *
 * Mem allocation on host
 * Init data
 *  for (RBLOCK){
 *      Mem allocation on device
 *      CopyDataFromHostToDevice
 *      Kernel computation
 *      CopyDataFromDeviceToHost
 *      FreeDevice
 *   }
 * FreeHost 
 ***********************************************************
 * Added parameters:
 *  -k number of kernel iterations
 *  -g GPU ckeck of redundant kernels (WARNING: must be 0 in case of UNHARDENED version)
 *  -b number of runs per block
 *
 ************************************************************
 *  UNHARD version: nn (-g 0)
 *  DMR version: nn_redundant
 *
 * default parameters
 * - friendly: filelist_64 -r 1000 -lat 30 -q 1 -w 850 -f 1 -s 850 -a 1 -k 1000 -g 0 -b 10
 * - heavy: filelist_64 -r 1000 -lat 30 -q 1 -w 1 -f 1 -s 1 -a 1 -k 1000 -g 0 -b 10
 *
 *
 ******************************************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <float.h>
#include <vector>
#include "cuda.h"

#define min( a, b )			a > b ? b : a
#define ceilDiv( a, b )		( a + b - 1 ) / b
#define print( x )			printf( #x ": %lu\n", (unsigned long) x )
#define DEBUG				false

#define DEFAULT_THREADS_PER_BLOCK 256

#define MAX_ARGS 10
#define REC_LENGTH 53 // size of a record in db
#define LATITUDE_POS 28	// character position of the latitude value in each record
#define OPEN 10000	// initial value of nearest neighbors

#ifdef TRIPLE
#define NUM_STREAMS 3
#else
#ifdef REDUNDANT
#define NUM_STREAMS 2
#endif
#endif


unsigned int runs_counter=0;
unsigned int runs_werror=0;
bool errors_flag=false;

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


//#ifdef REDUNDANT
bool float_equals(float a, float b, float epsilon = 0.001)
{
    return std::abs(a - b) < epsilon;
}
//#endif

struct timeval time_start;
struct timeval time_end;
long int get_time(struct timeval time_start, struct timeval time_end){
	long int r ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	return r;
}

#ifdef TIMING
	struct timeval Transfer1_start, Transfer1_end;
	struct timeval Transfer2_start, Transfer2_end;
#endif

  struct timeval Kernel1_start, Kernel1_end;

typedef struct latLong
{
  float lat;
  float lng;
} LatLong;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;


int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations);
void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN);
void printUsage();
int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int *w, int *f, int* s,
		     int* a, int* k, int* g, int *b);


__device__ int d_num_errors=0;
/**
 * Kernel
 * Computes differences between two vectors
 */
 __global__ void compute_difference(float *d_distances, float *d_distances_redundant, int numRecords)
{
  float epsilon = 0.001;
  int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
  if (globalId < numRecords) {
    float *dist=d_distances+globalId;
    float *distances_redundant=d_distances_redundant+globalId;
    if (std::abs((*dist) - (*distances_redundant)) < epsilon){
      //perform atomic add
      atomicAdd(&d_num_errors, 1);
    }
  }
} 

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng, int work_factor, int ITERACIONES_KERNEL)
{
  for (int i = 0; i < ITERACIONES_KERNEL; i++){
    int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
    for(int k = 0; k < work_factor; ++k){
      LatLong *latLong = d_locations+globalId;
      if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        // Versión con varias iteraciones por kernel
	if (i==0) {*dist=0;}
	*dist += (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
      }
      globalId += (gridDim.x * blockDim.x * gridDim.y * blockDim.y);
    }
  }// end niteraciones
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
  gettimeofday(&time_start, NULL);
  //int i=0;
  float lat, lng;
  int quiet=0,timing=0,platform=0,device=0;
  //int VERSION = 0;

  std::vector<Record> records;
  std::vector<LatLong> locations;
  char filename[100];
  int resultsCount=10;
  int ITERACIONES_KERNEL = 1;
  int COMPARACION_GPU = 0;
  int RBLOCK  = 1;
    
  int work_factor = 0;
  int thread_factor = 0;
  int work_factor_redundant = 0;
  int thread_factor_redundant = 0;
  long int TotalKernelExecutionTime = 0;
    // parse command line
  if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device, &work_factor,
		     &thread_factor, &work_factor_redundant,
		     &thread_factor_redundant, &ITERACIONES_KERNEL, &COMPARACION_GPU, &RBLOCK)) {
      printUsage();
      return 0;
    }
  
  printf("Hw: JetsonTX2, Pascal arch \n");
  printf("Test: NN\n");
  printf("RUNBLOCK: %d \n", RBLOCK);
  printf("Iteraciones Kernel: %d \n", ITERACIONES_KERNEL);
  printf("DMR_check en GPU: %d \n", COMPARACION_GPU);  
  fflush(stdout);
  
  
//////////////////////   SC Inicializa los datos ////////////////
  int numRecords = loadData(filename,records,locations);
  if (resultsCount > numRecords) resultsCount = numRecords;
/////////////////////////////////////////////////////////////////
    
  //Pointers to host memory
  float *distances;
  float *distances_golden;
  //Pointers to device memory
  LatLong *d_locations;
  float *d_distances;
/////// SC Calcula parámetros
  cudaDeviceProp deviceProp;
  HANDLE_ERROR( cudaGetDeviceProperties( &deviceProp, 0 ));
  HANDLE_ERROR( cudaDeviceSynchronize());
  unsigned long maxGridX = deviceProp.maxGridSize[0];
  unsigned long threadsPerBlock = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK )/thread_factor;
  unsigned long threadsPerBlock_redundant = min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK )/thread_factor_redundant;

  printf("Default (heavy) block configuration: (%d,%d,%d)\n",  min( deviceProp.maxThreadsPerBlock, DEFAULT_THREADS_PER_BLOCK )/thread_factor,1,1);

  unsigned long blocks = ceilDiv( numRecords, threadsPerBlock ); // extra threads will do nothing
  unsigned long gridY = ceilDiv( blocks, maxGridX );
  unsigned long gridX = ceilDiv( blocks, gridY );

  unsigned long blocks_redundant = ceilDiv( numRecords, threadsPerBlock_redundant ); // extra threads will do nothing
  unsigned long gridY_redundant = ceilDiv( blocks_redundant, maxGridX );
  unsigned long gridX_redundant = ceilDiv( blocks_redundant, gridY );
  // There will be no more than (gridY - 1) extra blocks
  dim3 gridDim( ceilDiv(gridX,work_factor) * thread_factor, gridY );
  dim3 gridDim_redundant( ceilDiv(gridX_redundant,work_factor_redundant) * thread_factor_redundant, gridY_redundant );
  printf("Default (heavy) grid configuration: (%lu,%lu,%d)\n", gridX/work_factor * thread_factor, gridY ,1);
  fflush(stdout);
	/**
	* Allocate memory on host and device
	*/
  distances = (float *)malloc(sizeof(float) * numRecords);
  distances_golden = (float *)malloc(sizeof(float) * numRecords);
  
///////////////////// SC Memory allocation on device must be done within the for(RBLOCK) loop
  // HANDLE_ERROR( cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords));
  // HANDLE_ERROR( cudaMalloc((void **) &d_distances,sizeof(float) * numRecords));
// #ifdef REDUNDANT
  // float *distances_redundant, *d_distances_redundant;
  // distances_redundant = (float *)malloc(sizeof(float) * numRecords);
  // HANDLE_ERROR( cudaMalloc((void **) &d_distances_redundant,sizeof(float) * numRecords));
// #endif
// #ifdef TRIPLE
  // float *distances_redundant2, *d_distances_redundant2;
  // distances_redundant2 = (float *)malloc(sizeof(float) * numRecords);
  // HANDLE_ERROR( cudaMalloc((void **) &d_distances_redundant2,sizeof(float) * numRecords));
// #endif

   /**
    * Transfer data from host to device
    */
/////////////////// SC Aquí empieza el loop /////////////////////////////////////////////
for (runs_counter=0; runs_counter < RBLOCK; runs_counter++){
   
// SC Allocate memory on device
  HANDLE_ERROR( cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords));
  HANDLE_ERROR( cudaMalloc((void **) &d_distances,sizeof(float) * numRecords));
#ifdef REDUNDANT
  float *distances_redundant, *d_distances_redundant;
  distances_redundant = (float *)malloc(sizeof(float) * numRecords);
  HANDLE_ERROR( cudaMalloc((void **) &d_distances_redundant,sizeof(float) * numRecords));
#endif
#ifdef TRIPLE
  float *distances_redundant2, *d_distances_redundant2;
  distances_redundant2 = (float *)malloc(sizeof(float) * numRecords);
  HANDLE_ERROR( cudaMalloc((void **) &d_distances_redundant2,sizeof(float) * numRecords));
#endif


  HANDLE_ERROR( cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice) );

    /**
    * Execute kernel
    */
    ////////////////////// VERSION REDUNDANT ///////////////////////////////////
#ifdef REDUNDANT
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; ++i)
    HANDLE_ERROR( cudaStreamCreate(&streams[i]) );

//#ifdef TIMING
  HANDLE_ERROR( cudaDeviceSynchronize() );
  gettimeofday(&Kernel1_start, NULL);
//#endif

    euclid<<< gridDim, threadsPerBlock, 0 , streams[0] >>>(d_locations,d_distances,numRecords,lat,lng, work_factor, ITERACIONES_KERNEL);

#ifdef SERIALIZE
  HANDLE_ERROR( cudaDeviceSynchronize() );
#endif
    euclid<<< gridDim_redundant, threadsPerBlock_redundant, 0 , streams[1] >>>(d_locations,d_distances_redundant,numRecords,lat,lng, work_factor_redundant, ITERACIONES_KERNEL);
#ifdef TRIPLE
#ifdef SERIALIZE
  HANDLE_ERROR( cudaDeviceSynchronize());
#endif
    euclid<<< gridDim, threadsPerBlock, 0 , streams[2] >>>(d_locations,d_distances_redundant2,numRecords,lat,lng, work_factor, ITERACIONES_KERNEL);
#endif

//#ifdef TIMING
  HANDLE_ERROR( cudaDeviceSynchronize());
  HANDLE_ERROR( cudaPeekAtLastError() );
  
   // kernel_check (); 
  gettimeofday(&Kernel1_end, NULL);
  TotalKernelExecutionTime += get_time(Kernel1_start, Kernel1_end);
//#endif

    ////////////////////// VERSION UNHARDENED ///////////////////////////////////
#else     

// Original version 
//#ifdef TIMING
  HANDLE_ERROR( cudaDeviceSynchronize());
	gettimeofday(&Kernel1_start, NULL);
//#endif
    euclid<<< gridDim, threadsPerBlock >>>(d_locations,d_distances,numRecords,lat,lng, work_factor, ITERACIONES_KERNEL);
    HANDLE_ERROR( cudaPeekAtLastError() );
//#ifdef TIMING
  HANDLE_ERROR( cudaDeviceSynchronize());
	gettimeofday(&Kernel1_end, NULL);
  TotalKernelExecutionTime += get_time(Kernel1_start, Kernel1_end);
  // SC added 
  // golden is initialized in the first iteration or in case o errors
  if (runs_counter==0 || errors_flag==true){ 
    HANDLE_ERROR( cudaMemcpy( distances_golden, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost ) );
    errors_flag=false; 
  }
  
//#endif

#endif

  HANDLE_ERROR( cudaDeviceSynchronize() );

    /**
    * SC Copy Results
    */
  
  if (!COMPARACION_GPU){
    //Copy data from device memory to host memory
    HANDLE_ERROR( cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost ) );
    
#ifdef REDUNDANT
#ifdef TIMING
    gettimeofday(&Transfer1_start, NULL);
#endif
    HANDLE_ERROR( cudaMemcpy( distances_redundant, d_distances_redundant, sizeof(float)*numRecords, cudaMemcpyDeviceToHost ) );

#ifdef TRIPLE
    HANDLE_ERROR( cudaMemcpy( distances_redundant2, d_distances_redundant2, sizeof(float)*numRecords, cudaMemcpyDeviceToHost ) );
#endif

#ifdef TIMING
    gettimeofday(&Transfer1_end, NULL);
#endif
#endif
  }

    /**
    * SC Check Results
    */
    
    
/////////// SC Checking REDUNDANT /////////////////////////////////////////////////////
#ifdef REDUNDANT
#ifdef TIMING
    struct timeval time_compare_begin1;
    struct timeval time_compare_end1;
	
    gettimeofday(&time_compare_begin1, NULL);
#endif


bool correct = true;
int num_errors_cpur=0;
int num_errors_cput=0;
  typeof(d_num_errors) num_errors = 0;
if (COMPARACION_GPU){ //start comparison in GPU
  compute_difference<<< gridDim, threadsPerBlock >>>(d_distances, d_distances_redundant, numRecords);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );
#ifdef TRIPLE
  compute_difference<<< gridDim, threadsPerBlock >>>(d_distances, d_distances_redundant2, numRecords);
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );
#endif

#ifdef TIMING
  gettimeofday(&time_compare_end1, NULL);	
  gettimeofday(&Transfer1_start, NULL);
#endif

  //Pass the result to the HOST
  HANDLE_ERROR(cudaMemcpyFromSymbol(&num_errors, d_num_errors, sizeof(num_errors), 0, cudaMemcpyDeviceToHost));
  HANDLE_ERROR( cudaDeviceSynchronize() );
  //Update correct
  correct = (num_errors == 0);

#ifdef TIMING
  gettimeofday(&Transfer1_end, NULL);
#endif
  }//end comparison in GPU FOR REDUNDANT
else{ //start comparison in CPU
    // for(int i = 0; i < numRecords && correct; ++i){
    for(int i = 0; i < numRecords; i++){
      correct = float_equals(distances[i], distances_redundant[i]);
      if (correct==false){ num_errors_cpur++;}
    }
#ifdef TRIPLE
    // for(int i = 0; i < numRecords && correct; ++i){
    for(int i = 0; i < numRecords; i++){
      correct = float_equals(distances[i], distances_redundant2[i]);
      if (correct==false){ num_errors_cput++;}
    }
#endif

#ifdef TIMING
    gettimeofday(&time_compare_end1, NULL);	
#endif

 } //end comparison in CPU FOR REDUNDANT

#ifdef TIMING
    printf("Result transfer time1: %ld us\n", get_time(Transfer1_start, Transfer1_end));
    printf("Comparison time1: %ld us\n", get_time(time_compare_begin1, time_compare_end1));
    fflush(stdout);
#endif

    if (num_errors_cpur==0 && num_errors_cput==0)
        printf("OK\n"); 
    else{
        printf("ERROR detected\n");
        if (COMPARACION_GPU) {
          printf("%d ERRORS detected\n", num_errors);
        }
        else { // SC necesita modificaciones para el TRIPLE
          printf("%d ERRORS detected\n", num_errors_cpur);
        }
        //SC  si hay errores inicializamos los datos
        num_errors_cpur=0;
        num_errors_cput=0;

        numRecords = loadData(filename,records,locations);
        runs_werror++;
        //exit(0);
    }
    fflush(stdout);

    //Clean up
    HANDLE_ERROR( cudaFree(d_distances_redundant) );
    free(distances_redundant);

    #ifdef TRIPLE
    HANDLE_ERROR( cudaFree(d_distances_redundant2) );
    free(distances_redundant2);
    #endif

    for ( int i = 0; i < NUM_STREAMS; i++)
      HANDLE_ERROR( cudaStreamDestroy(streams[i]) );

/////////// SC Checking  UNHARD /////////////////////////////////////////////////
#else  
    bool correct_unh = true;
    int num_errors_unh=0;
    for(int i = 0; i < numRecords ; i++){
      correct_unh = float_equals(distances[i], distances_golden[i]);
       if (correct_unh==false){ num_errors_unh++;}
    }
    
    if (num_errors_unh == 0)
        printf("OK\n"); 
    else{
      printf("%d ERRORS detected\n", num_errors_unh);

        //SC  si hay errores inicializamos los datos
        num_errors_unh=0; 
        errors_flag=true; 
        numRecords = loadData(filename,records,locations);
        runs_werror++;
        //exit(0);
    }
    fflush(stdout);

#endif


#ifdef PRINT_OUTPUT
    for(int i = 0; i < numRecords; ++i)
        printf("%f", distances[i]);
    printf("\n");
#endif

// SC esto sobra  /////////////////////////////////////////////////////////////////////////////////
	// find the resultsCount least distances
    // findLowest(records,distances,numRecords,resultsCount);

    // print out results
    // if (!quiet)
    // for(i=0;i<resultsCount;i++) {
      // printf("%s --> Distance=%f\n",records[i].recString,records[i].distance);
    // }
/////////////////////////////////////////////////////////////////////////////////////////////////////
    // Este free solo se hace al final del bucle for (RBLOCK)
    //free(distances);
  //Free memory
  HANDLE_ERROR( cudaFree(d_locations) );
  HANDLE_ERROR( cudaFree(d_distances) );

  gettimeofday(&time_end, NULL);
  long int TotalExecutionTime = get_time(time_start, time_end);
  printf("Execution time: %ld us\n", TotalExecutionTime);
  long int Kernel1;
  //Kernel1 = get_time(Kernel1_start, Kernel1_end);
  //long int TotalKernelExecutionTime = Kernel1;
  printf("Time for CUDA kernels:\t%ld us\n",TotalKernelExecutionTime);

#ifdef TIMING
  //// SC?? no entiendo este cálculo (corresponde a la ejecución del último run??)
  Kernel1 = get_time(Kernel1_start, Kernel1_end);
  TotalKernelExecutionTime = Kernel1;
  printf("Total Kernel Execution Time: %ld\n", TotalKernelExecutionTime);

#ifndef REDUNDANT
  long int TotalTransferTime = 0;
  printf("Transfer Time: %ld, Kernel Time: %ld, Execution Time: %ld\n", TotalTransferTime, TotalKernelExecutionTime, TotalExecutionTime);
  printf("GPU time (Transfers + Kernels): %f%%\n", (TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100);
  printf("CPU time (rest): %f%%\n", 100 - ((TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100));
#endif
fflush(stdout);
#endif

////////////////////// SC FIN DEL BUCLE ///////////////////////////////////////////////////
  }
  // free host memory
free(distances);
free(distances_golden);

printf("TEST_CHECK:%u;RUNS_WERROR:%d\n", RBLOCK, runs_werror);
fflush(stdout);
}



int loadData(char *filename,std::vector<Record> &records,std::vector<LatLong> &locations){
    FILE   *flist,*fp;
	int    i=0;
	char dbname[64];
	int recNum=0;

    /**Main processing **/

    flist = fopen(filename, "r");
	while(!feof(flist)) {
		/**
		* Read in all records of length REC_LENGTH
		* If this is the last file in the filelist, then done
		* else open next file to be read next iteration
		*/
		if(fscanf(flist, "%s\n", dbname) != 1) {
            fprintf(stderr, "error reading filelist\n");
            exit(0);
        }
        fp = fopen(dbname, "r");
        if(!fp) {
            printf("error opening a db\n");
            exit(1);
        }
        // read each record
        while(!feof(fp)){
            Record record;
            LatLong latLong;
            fgets(record.recString,49,fp);
            fgetc(fp); // newline
            if (feof(fp)) break;

            // parse for lat and long
            char substr[6];

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+28);
            substr[5] = '\0';
            latLong.lat = atof(substr);

            for(i=0;i<5;i++) substr[i] = *(record.recString+i+33);
            substr[5] = '\0';
            latLong.lng = atof(substr);

            locations.push_back(latLong);
            records.push_back(record);
            recNum++;
        }
        fclose(fp);
    }
    fclose(flist);
//    for(i=0;i<rec_count*REC_LENGTH;i++) printf("%c",sandbox[i]);
    return recNum;
}

void findLowest(std::vector<Record> &records,float *distances,int numRecords,int topN){
  int i,j;
  float val;
  int minLoc;
  Record *tempRec;
  float tempDist;

  for(i=0;i<topN;i++) {
    minLoc = i;
    for(j=i;j<numRecords;j++) {
      val = distances[j];
      if (val < distances[minLoc]) minLoc = j;
    }
    // swap locations and distances
    tempRec = &records[i];
    records[i] = records[minLoc];
    records[minLoc] = *tempRec;

    tempDist = distances[i];
    distances[i] = distances[minLoc];
    distances[minLoc] = tempDist;

    // add distance to the min we just found
    records[i].distance = distances[i];
  }
}

int parseCommandline(int argc, char *argv[], char* filename,int *r,float *lat,float *lng,
                     int *q, int *t, int *p, int *d, int *w, int *f, int *s,
		     int *a, int *k, int *g, int *b){
    int i;
    if (argc < 2) return 1; // error
    strncpy(filename,argv[1],100);
    char flag;

    for(i=1;i<argc;i++) {
      if (argv[i][0]=='-') {// flag
        flag = argv[i][1];
          switch (flag) {
            case 'r': // number of results
              i++;
              *r = atoi(argv[i]);
              break;
            case 'l': // lat or lng
              if (argv[i][2]=='a') {//lat
                *lat = atof(argv[i+1]);
              }
              else {//lng
                *lng = atof(argv[i+1]);
              }
              i++;
              break;
            case 'h': // help
              return 1;
            case 'q': // quiet
              *q = 1;
              break;
            case 't': // timing
              *t = 1;
              break;
            case 'p': // platform
              i++;
              *p = atoi(argv[i]);
              break;
            case 'd': // device
              i++;
              *d = atoi(argv[i]);
              break;
            case 'w':
              i++;
              *w = atoi(argv[i]);
              break;
            case 'f':
              i++;
              *f = atoi(argv[i]);
              break;
            case 's':
              i++;
              *s = atoi(argv[i]);
              break;
            case 'a':
              i++;
              *a = atoi(argv[i]);
              break;
            case 'k'://kernel iterations
              i++;
              *k = atoi(argv[i]);
              break;
            case 'g'://comparacion por kernel
              i++;
              //*g = 1; 
              // SC modificado para que lo lea como argumento
              *g = atoi(argv[i]);
              break;
            // SC añadido para parametrizar el num de runs por del bloque
            case 'b'://comparacion por kernel
              i++;
              *b = atoi(argv[i]);
              break;

        }
      }
    }
    if ((*d >= 0 && *p<0) || (*p>=0 && *d<0)) // both p and d must be specified if either are specified
      return 1;
    return 0;
}

void printUsage(){
  printf("Nearest Neighbor Usage\n");
  printf("\n");
  printf("nearestNeighbor [filename] -r [int] -lat [float] -lng [float] [-hqt] [-p [int] -d [int]]\n");
  printf("\n");
  printf("example:\n");
  printf("$ ./nearestNeighbor filelist.txt -r 5 -lat 30 -lng 90\n");
  printf("\n");
  printf("filename     the filename that lists the data input files\n");
  printf("-r [int]     the number of records to return (default: 10)\n");
  printf("-lat [float] the latitude for nearest neighbors (default: 0)\n");
  printf("-lng [float] the longitude for nearest neighbors (default: 0)\n");
  printf("\n");
  printf("-h, --help   Display the help file\n");
  printf("-q           Quiet mode. Suppress all text output.\n");
  printf("-t           Print timing information.\n");
  printf("\n");
  printf("-p [int]     Choose the platform (must choose both platform and device)\n");
  printf("-d [int]     Choose the device (must choose both platform and device)\n");
  printf("\n");
  printf("-k [int]     Choose the number of kernel iterations\n");
  printf("-g [int]     If 1, the comparison will be made by the gpu kernel\n");
  printf("-b [int]     choose the number of runs\n");
  printf("\n");
  printf("-v [int([1 - 3])]     Choose the Version (of grid configuration) to be used by the kernels\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
