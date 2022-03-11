/*
 * nn.cu
 * Nearest Neighbor
 * SC modified for radiation tests
 */

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

#define RBLOCK  10

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


#ifdef REDUNDANT
bool float_equals(float a, float b, float epsilon = 0.001)
{
    return std::abs(a - b) < epsilon;
}
#endif

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
		     int* a);

/**
* Kernel
* Executed on GPU
* Calculates the Euclidean distance from each record in the database to the target position
*/
__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords,float lat, float lng, int work_factor)
{
  int globalId = gridDim.x * blockDim.x * blockIdx.y + blockDim.x * blockIdx.x + threadIdx.x;
  //int globalId = (blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x) * work_factor; // more efficient
  for(int k = 0; k < work_factor; ++k){
    LatLong *latLong = d_locations+globalId;
    if (globalId < numRecords) {
        float *dist=d_distances+globalId;
        *dist = (float)sqrt((lat-latLong->lat)*(lat-latLong->lat)+(lng-latLong->lng)*(lng-latLong->lng));
	  }
    //globalId++;
    //globalId += work_factor;
    globalId += (gridDim.x * blockDim.x * gridDim.y * blockDim.y);
    //globalId = (blockDim.x * blockIdx.x + threadIdx.x);
  }
}

/**
* This program finds the k-nearest neighbors
**/

int main(int argc, char* argv[])
{
  gettimeofday(&time_start, NULL);
	int    i=0;
	float lat, lng;
	int quiet=0,timing=0,platform=0,device=0;
  //int VERSION = 0;

  std::vector<Record> records;
	std::vector<LatLong> locations;
	char filename[100];
	int resultsCount=10;
    


  int work_factor = 0;
  int thread_factor = 0;
  int work_factor_redundant = 0;
  int thread_factor_redundant = 0;
    // parse command line
    if (parseCommandline(argc, argv, filename,&resultsCount,&lat,&lng,
                     &quiet, &timing, &platform, &device, &work_factor,
		     &thread_factor, &work_factor_redundant,
		     &thread_factor_redundant)) {
      printUsage();
      return 0;
    }
//////////////////////   SC Inicializa los datos ////////////////
    int numRecords = loadData(filename,records,locations);
    if (resultsCount > numRecords) resultsCount = numRecords;
/////////////////////////////////////////////////////////////////
    
    //Pointers to host memory
	float *distances;
	//Pointers to device memory
	LatLong *d_locations;
	float *d_distances;
/////// SC Calcula parámetros
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties( &deviceProp, 0 );
  cudaThreadSynchronize();
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

	/**
	* Allocate memory on host and device
	*/
	distances = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_locations,sizeof(LatLong) * numRecords);
	cudaMalloc((void **) &d_distances,sizeof(float) * numRecords);
#ifdef REDUNDANT
  float *distances_redundant, *d_distances_redundant;
	distances_redundant = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_distances_redundant,sizeof(float) * numRecords);
#endif
#ifdef TRIPLE
  float *distances_redundant2, *d_distances_redundant2;
	distances_redundant2 = (float *)malloc(sizeof(float) * numRecords);
	cudaMalloc((void **) &d_distances_redundant2,sizeof(float) * numRecords);
#endif

   /**
    * Transfer data from host to device
    */
/////////////////// SC Aquí empieza el loop /////////////////////////////////////////////
for (runs_counter=0; runs_counter < RBLOCK; runs_counter++){

    cudaMemcpy( d_locations, &locations[0], sizeof(LatLong) * numRecords, cudaMemcpyHostToDevice);

    /**
    * Execute kernel
    */
#ifdef REDUNDANT
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

//#ifdef TIMING
  cudaThreadSynchronize();
	gettimeofday(&Kernel1_start, NULL);
//#endif

    euclid<<< gridDim, threadsPerBlock, 0 , streams[0] >>>(d_locations,d_distances,numRecords,lat,lng, work_factor);
#ifdef SERIALIZE
    cudaThreadSynchronize();
#endif
    euclid<<< gridDim_redundant, threadsPerBlock_redundant, 0 , streams[1] >>>(d_locations,d_distances_redundant,numRecords,lat,lng, work_factor_redundant);
#ifdef TRIPLE
#ifdef SERIALIZE
    cudaThreadSynchronize();
#endif
    euclid<<< gridDim, threadsPerBlock, 0 , streams[2] >>>(d_locations,d_distances_redundant2,numRecords,lat,lng, work_factor);
#endif

//#ifdef TIMING
  cudaThreadSynchronize();
  gettimeofday(&Kernel1_end, NULL);
//#endif

#else

// Original version 
//#ifdef TIMING
  cudaThreadSynchronize();
	gettimeofday(&Kernel1_start, NULL);
//#endif
    euclid<<< gridDim, threadsPerBlock >>>(d_locations,d_distances,numRecords,lat,lng, work_factor);
//#ifdef TIMING
	cudaThreadSynchronize();
	gettimeofday(&Kernel1_end, NULL);
//#endif

#endif
    cudaThreadSynchronize();

  //Copy data from device memory to host memory
  cudaMemcpy( distances, d_distances, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );

#ifdef REDUNDANT
#ifdef TIMING
  gettimeofday(&Transfer1_start, NULL);
#endif
    cudaMemcpy( distances_redundant, d_distances_redundant, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );

#ifdef TRIPLE
    cudaMemcpy( distances_redundant2, d_distances_redundant2, sizeof(float)*numRecords, cudaMemcpyDeviceToHost );
#endif

#ifdef TIMING
  gettimeofday(&Transfer1_end, NULL);
#endif

#ifdef TIMING
    struct timeval time_compare_begin1;
    struct timeval time_compare_end1;
	
    gettimeofday(&time_compare_begin1, NULL);
#endif

// SC REaliza el check //////////////////////////////////////////////////////////////////////////////
    bool correct = true;
    for(int i = 0; i < numRecords && correct; ++i)
        correct = float_equals(distances[i], distances_redundant[i]);
#ifdef TRIPLE
    for(int i = 0; i < numRecords && correct; ++i)
        correct = float_equals(distances[i], distances_redundant2[i]);
#endif

#ifdef TIMING
    gettimeofday(&time_compare_end1, NULL);	
    printf("Result transfer time1: %ld us\n", get_time(Transfer1_start, Transfer1_end));
    printf("Comparison time1: %ld us\n", get_time(time_compare_begin1, time_compare_end1));

#endif

    if (correct)
        printf("OK\n");
    else{
        printf("ERROR detected\n");
        
        //SC 
        numRecords = loadData(filename,records,locations);
        runs_werror++;
        //exit(0);
    }
/////////// SC Fin el checking ///////////////////////////
    //Clean up
    cudaFree(d_distances_redundant);
    free(distances_redundant);

    #ifdef TRIPLE
    cudaFree(d_distances_redundant2);
    free(distances_redundant2);
    #endif

    for ( int i = 0; i < NUM_STREAMS; i++)
        cudaStreamDestroy(streams[i]);

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
    free(distances);
    //Free memory
	cudaFree(d_locations);
	cudaFree(d_distances);

	gettimeofday(&time_end, NULL);
	long int TotalExecutionTime = get_time(time_start, time_end);
	printf("Execution time: %ld us\n", TotalExecutionTime);
  long int Kernel1;
  Kernel1 = get_time(Kernel1_start, Kernel1_end);
  long int TotalKernelExecutionTime = Kernel1;
  printf("Time for CUDA kernels:\t%ld us\n",TotalKernelExecutionTime);

#ifdef TIMING
	Kernel1 = get_time(Kernel1_start, Kernel1_end);
	TotalKernelExecutionTime = Kernel1;
    printf("Total Kernel Execution Time: %ld\n", TotalKernelExecutionTime);

#ifndef REDUNDANT
  long int TotalTransferTime = 0;
	printf("Transfer Time: %ld, Kernel Time: %ld, Execution Time: %ld\n", TotalTransferTime, TotalKernelExecutionTime, TotalExecutionTime);
	printf("GPU time (Transfers + Kernels): %f%%\n", (TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100);
	printf("CPU time (rest): %f%%\n", 100 - ((TotalTransferTime+TotalKernelExecutionTime)/(TotalExecutionTime*1.0)*100));
#endif

#endif

////////////////////// SC FIN DEL BUCLE ///////////////////////////////////////////////////
}

printf("TEST_CHECK:%u;RUNS_WERROR:%d\n", RBLOCK, runs_werror);

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
		     int *a){
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
  printf("\n");
  printf("-v [int([1 - 3])]     Choose the Version (of grid configuration) to be used by the kernels\n");
  printf("Notes: 1. The filename is required as the first parameter.\n");
  printf("       2. If you declare either the device or the platform,\n");
  printf("          you must declare both.\n\n");
}
