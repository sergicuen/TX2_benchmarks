//====================================================================================================100
//		DEFINE / INCLUDE
//====================================================================================================100

#include "kernel_fin_2.cu"
#include "kernel_ecc_2.cu"
#include "kernel_cam_2.cu"
#include "kernel_2.cu"
#include "embedded_fehlberg_7_8_2.cu"
#include "solver_2.cu"
#include <assert.h>
#include<sys/time.h>

#ifdef TIMING
	extern struct timeval RedTransfer1_start, RedTransfer1_end;
	extern struct timeval RedTransfer2_start, RedTransfer2_end;
	extern struct timeval Transfer1_start, Transfer1_end;
	extern struct timeval Transfer2_start, Transfer2_end;
	extern struct timeval Kernel1_start, Kernel1_end;
#endif
#ifdef TRIPLE
#define NUM_STREAMS 3
#else
#ifdef REDUNDANT
#define NUM_STREAMS 2
#endif
#endif

#ifdef REDUNDANT
bool float_equals(float a, float b, float epsilon = 0.001)
{
    return std::abs(a - b) < epsilon;
}
#endif
long int get_times(struct timeval time_start, struct timeval time_end){
	long int r ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec));
	return r;
}
//====================================================================================================100
//		MAIN FUNCTION
//====================================================================================================100

int work_2(	int xmax,
					int workload){

	//================================================================================80
	//		VARIABLES
	//================================================================================80

	//============================================================60
	//		TIME
	//============================================================60

	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;

	time0 = get_time();

	//============================================================60
	//		COUNTERS, POINTERS
	//============================================================60

	long memory;
	int i;
	int pointer;

	//============================================================60
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//============================================================60

	fp* y;
	fp* d_y;

	long y_mem;

	fp* x;
	fp* d_x;

	long x_mem;

	fp* params;
	fp* d_params;

	int params_mem;
#ifdef REDUNDANT	
	fp* y_redundant;
	fp* d_y_redundant;

	fp* x_redundant;
	fp* d_x_redundant;

	fp* d_params_redundant;
#endif
#ifdef TRIPLE
	fp* y_redundant2;
	fp* d_y_redundant2;

	fp* x_redundant2;
	fp* d_x_redundant2;

	fp* d_params_redundant2;
#endif
	//============================================================60
	//		TEMPORARY SOLVER VARIABLES
	//============================================================60

	fp* d_com;
	int com_mem;

	fp* d_err;
	int err_mem;

	fp* d_scale;
	int scale_mem;

	fp* d_yy;
	int yy_mem;

	fp* d_initvalu_temp;
	int initvalu_temp_mem;

	fp* d_finavalu_temp;
	int finavalu_temp_mem;

#ifdef REDUNDANT
	fp* d_com_redundant;
	fp* d_err_redundant;
	fp* d_scale_redundant;
	fp* d_yy_redundant;
	fp* d_initvalu_temp_redundant;
	fp* d_finavalu_temp_redundant;
#endif
  
#ifdef TRIPLE
	fp* d_com_redundant2;
	fp* d_err_redundant2;
	fp* d_scale_redundant2;
	fp* d_yy_redundant2;
	fp* d_initvalu_temp_redundant2;
	fp* d_finavalu_temp_redundant2;
#endif
	//============================================================60
	//		CUDA KERNELS EXECUTION PARAMETERS
	//============================================================60

	dim3 threads;
	dim3 blocks;
	int blocks_x;

	time1 = get_time();

	//================================================================================80
	// 	ALLOCATE MEMORY
	//================================================================================80

	//============================================================60
	//		MEMORY CHECK
	//============================================================60

	memory = workload*(xmax+1)*EQUATIONS*4;
	if(memory>1000000000){
		printf("ERROR: trying to allocate more than 1.0GB of memory, decrease workload and span parameters or change memory parameter\n");
		return 0;
	}

	//============================================================60
	// 	ALLOCATE ARRAYS
	//============================================================60

	//========================================40
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//========================================40

	y_mem = workload * (xmax+1) * EQUATIONS * sizeof(fp);
	y= (fp *) malloc(y_mem);
#ifdef REDUNDANT
	y_redundant= (fp *) malloc(y_mem);
#endif
#ifdef TRIPLE
	y_redundant2= (fp *) malloc(y_mem);
#endif
	cudaMalloc((void **)&d_y, y_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_y_redundant, y_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_y_redundant2, y_mem);
#endif
	x_mem = workload * (xmax+1) * sizeof(fp);
	x= (fp *) malloc(x_mem);
#ifdef REDUNDANT
	x_redundant= (fp *) malloc(x_mem);
#endif
#ifdef TRIPLE
	x_redundant2= (fp *) malloc(x_mem);
#endif
	cudaMalloc((void **)&d_x, x_mem);

#ifdef REDUNDANT
	cudaMalloc((void **)&d_x_redundant, x_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_x_redundant2, x_mem);
#endif
	params_mem = workload * PARAMETERS * sizeof(fp);
	params= (fp *) malloc(params_mem);
	cudaMalloc((void **)&d_params, params_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_params_redundant, params_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_params_redundant2, params_mem);
#endif
	//========================================40
	//		TEMPORARY SOLVER VARIABLES
	//========================================40

	com_mem = workload * 3 * sizeof(fp);
	cudaMalloc((void **)&d_com, com_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_com_redundant, com_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_com_redundant2, com_mem);
#endif

	err_mem = workload * EQUATIONS * sizeof(fp);
	cudaMalloc((void **)&d_err, err_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_err_redundant, err_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_err_redundant2, err_mem);
#endif

	scale_mem = workload * EQUATIONS * sizeof(fp);
	cudaMalloc((void **)&d_scale, scale_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_scale_redundant, scale_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_scale_redundant2, scale_mem);
#endif

	yy_mem = workload * EQUATIONS * sizeof(fp);
	cudaMalloc((void **)&d_yy, yy_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_yy_redundant, yy_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_yy_redundant2, yy_mem);
#endif

	initvalu_temp_mem = workload * EQUATIONS * sizeof(fp);
	cudaMalloc((void **)&d_initvalu_temp, initvalu_temp_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_initvalu_temp_redundant, initvalu_temp_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_initvalu_temp_redundant2, initvalu_temp_mem);
#endif

	finavalu_temp_mem = workload * 13* EQUATIONS * sizeof(fp);
	cudaMalloc((void **)&d_finavalu_temp, finavalu_temp_mem);
#ifdef REDUNDANT
	cudaMalloc((void **)&d_finavalu_temp_redundant, finavalu_temp_mem);
#endif
#ifdef TRIPLE
	cudaMalloc((void **)&d_finavalu_temp_redundant2, finavalu_temp_mem);
#endif

	time2 = get_time();

	//================================================================================80
	// 	READ FROM FILES OR SET INITIAL VALUES
	//================================================================================80

	//========================================40
	//		X
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i * (xmax+1) + 0;
		x[pointer] = 0;
    }
	
	//========================================40
	//		Y
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i*((xmax+1)*EQUATIONS) + 0*(EQUATIONS);
		read("data/y.txt",
					&y[pointer],
					91,
					1,
					0);
	}


	//========================================40
	//		PARAMS
	//========================================40

	for(i=0; i<workload; i++){
		pointer = i*PARAMETERS;
		read("data/params.txt",
					&params[pointer],
					18,
					1,
					0);
	}
#ifdef TIMING
	gettimeofday(&Transfer1_start, NULL);
#endif
	cudaMemcpy(d_x, x, x_mem, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, y_mem, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, params, params_mem, cudaMemcpyHostToDevice);
#ifdef TIMING
	gettimeofday(&Transfer1_end, NULL);
#endif



#ifdef REDUNDANT
  memcpy((void *)x_redundant, (void *)x, (size_t)x_mem);
  memcpy((void *)y_redundant, (void *)y, (size_t)y_mem);

#ifdef TRIPLE
  memcpy((void *)x_redundant2, (void *)x, x_mem);
  memcpy((void *)y_redundant2, (void *)y, y_mem);
#endif

#ifdef TIMING
	gettimeofday(&RedTransfer1_start, NULL);
#endif
	cudaMemcpy(d_x_redundant, d_x, x_mem, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_y_redundant, d_y, y_mem, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_params_redundant, d_params, params_mem, cudaMemcpyDeviceToDevice);
	
#ifdef TRIPLE
	cudaMemcpy(d_x_redundant2, d_x, x_mem, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_y_redundant2, d_y, y_mem, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_params_redundant2, d_params, params_mem, cudaMemcpyDeviceToDevice);
#endif

#ifdef TIMING
	gettimeofday(&RedTransfer1_end, NULL);
#endif
#endif
	time3 = get_time();

	//================================================================================80
	//		EXECUTION IF THERE ARE MANY WORKLOADS
	//================================================================================80

	if(workload == 1){
		threads.x = 32;																			// define the number of threads in the block
		threads.y = 1;
		blocks.x = 4;																				// define the number of blocks in the grid
		blocks.y = 1;
	}
	else{
		threads.x = NUMBER_THREADS;												// define the number of threads in the block
		threads.y = 1;
		blocks_x = workload/threads.x;
		if (workload % threads.x != 0){												// compensate for division remainder above by adding one grid
			blocks_x = blocks_x + 1;
		}
		blocks.x = blocks_x;																	// define the number of blocks in the grid
		blocks.y = 1;
	}
#ifdef REDUNDANT
  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++)
    cudaStreamCreate(&streams[i]);
#endif

#ifdef REDUNDANT
	assert(d_x != d_x_redundant);
	assert(d_y != d_y_redundant);
#ifdef TRIPLE
	assert(d_x != d_x_redundant2);
	assert(d_y != d_y_redundant2);
#endif
#endif

#ifdef TIMING
    cudaDeviceSynchronize();
	gettimeofday(&Kernel1_start, NULL);
#endif

#ifdef REDUNDANT
	solver_2<<<blocks, threads, 0, streams[0] >>>(workload, xmax, d_x, d_y, d_params, d_com, d_err, d_scale, d_yy, d_initvalu_temp, d_finavalu_temp);
#ifdef SERIALIZE
    cudaDeviceSynchronize();
#endif
#ifdef REPLICATE_DATA
	solver_2<<<blocks, threads, 0, streams[1] >>>(workload, xmax, d_x_redundant, d_y_redundant, d_params_redundant,	d_com_redundant, d_err_redundant, d_scale_redundant, d_yy_redundant, d_initvalu_temp_redundant, d_finavalu_temp_redundant);
#else
	solver_2<<<blocks, threads, 0, streams[1] >>>(workload, xmax, d_x_redundant, d_y_redundant, d_params_redundant, d_com_redundant, d_err_redundant, d_scale, d_yy, d_initvalu_temp_redundant, d_finavalu_temp_redundant);
#endif//REPLICATE_DATA

    ///////////
#ifdef TRIPLE
#ifdef SERIALIZE
    cudaDeviceSynchronize();
#endif
#ifdef REPLICATE_DATA
	solver_2<<<blocks, threads, 0, streams[2] >>>(workload, xmax, d_x_redundant2, d_y_redundant2, d_params_redundant2,	d_com_redundant2, d_err_redundant2, d_scale_redundant2, d_yy_redundant2, d_initvalu_temp_redundant2, d_finavalu_temp_redundant2);
#else
	solver_2<<<blocks, threads, 0, streams[2] >>>(workload, xmax, d_x_redundant2, d_y_redundant2, d_params_redundant2,	d_com_redundant2, d_err_redundant2, d_scale, d_yy, d_initvalu_temp_redundant2, d_finavalu_temp_redundant2);
#endif//REPLICATE_DATA
#endif//TRIPLE

////////



#else

	solver_2<<<blocks, threads>>>(workload, xmax, d_x, d_y, d_params, d_com, d_err, d_scale, d_yy, d_initvalu_temp, d_finavalu_temp);
#endif//REDUNDANT

#ifdef TIMING
	cudaDeviceSynchronize();
	gettimeofday(&Kernel1_end, NULL);
#endif
	// cudaThreadSynchronize();
	// printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
#ifdef REDUNDANT
  for (int i = 0;i < NUM_STREAMS; i++)
    cudaStreamSynchronize(streams[i]);
#endif
	time4 = get_time();

	//================================================================================80
	//		COPY DATA BACK TO CPU
	//================================================================================80
#ifdef TIMING
	gettimeofday(&Transfer2_start, NULL);
#endif
	cudaMemcpy(x, d_x, x_mem, cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, y_mem, cudaMemcpyDeviceToHost);
#ifdef TIMING
	gettimeofday(&Transfer2_end, NULL);
#endif

#ifdef REDUNDANT

#ifdef TIMING
	gettimeofday(&RedTransfer2_start, NULL);
#endif

	cudaMemcpy(x_redundant, d_x_redundant, x_mem, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_redundant, d_y_redundant, y_mem, cudaMemcpyDeviceToHost);

#ifdef TRIPLE
	cudaMemcpy(x_redundant2, d_x_redundant2, x_mem, cudaMemcpyDeviceToHost);
	cudaMemcpy(y_redundant2, d_y_redundant2, y_mem, cudaMemcpyDeviceToHost);
#endif

#ifdef TIMING
	gettimeofday(&RedTransfer2_end, NULL);
#endif

#endif
	time5 = get_time();

	//================================================================================80
	//		PRINT RESULTS (ENABLE SELECTIVELY FOR TESTING ONLY)
	//================================================================================80

	// int j, k;

	// for(i=0; i<workload; i++){
		// printf("WORKLOAD %d:\n", i);
		// for(j=0; j<(xmax+1); j++){
			// printf("\tTIME %d:\n", j);
			// for(k=0; k<EQUATIONS; k++){
				// printf("\t\ty[%d][%d][%d]=%13.10f\n", i, j, k, y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
			// }
		// }
	// }

	// for(i=0; i<workload; i++){
		// printf("WORKLOAD %d:\n", i);
		// for(j=0; j<(xmax+1); j++){
			// printf("\tTIME %d:\n", j);
				// printf("\t\tx[%d][%d]=%13.10f\n", i, j, x[i * (xmax+1) + j]);
		// }
	// }

	//================================================================================80
	//		CHECK RESULTS
	//================================================================================80
#ifdef REDUNDANT

#ifdef TIMING
  struct timeval time_compare1;
  struct timeval time_compare2;
  cudaDeviceSynchronize();
  gettimeofday(&time_compare1, NULL);
#endif

  int j, k;
  bool correct = true;
	for(i=0; i<workload and correct; i++){
		for(j=0; j<(xmax+1) and correct; j++){
			for(k=0; k<EQUATIONS and correct; k++){
        #ifdef PRINT_OUTPUT
        printf("%f == %f\n", y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k], y_redundant[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
        #endif
        correct = float_equals(y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k], y_redundant[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
      }
    }
  }
  #ifdef PRINT_OUTPUT
  printf("END\n");
  #endif

#ifdef TRIPLE
	for(i=0; i<workload and correct; i++){
		for(j=0; j<(xmax+1) and correct; j++){
			for(k=0; k<EQUATIONS and correct; k++){
        #ifdef PRINT_OUTPUT
        printf("%f\n", y_redundant2[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
        #endif
        correct = float_equals(y[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k], y_redundant2[i*((xmax+1)*EQUATIONS) + j*(EQUATIONS)+k]);
      }
    }
  }

  #ifdef PRINT_OUTPUT
  printf("\n");
  #endif

#endif//TRIPLE
  if (correct)
    printf("Redundant executions produced the same results !!!\n");
  else{
    printf("ERROR: Redundant executions produced different results(1)!!!!!\n");
    exit(0);
  }
  

  for(i=0; i<workload and correct; i++){
    for(j=0; j<(xmax+1) and correct; j++){
      correct = float_equals(x[i * (xmax+1) + j], x_redundant[i * (xmax+1) + j]);
#ifdef PRINT_OUTPUT
      printf("%f", x[i * (xmax+1) + j]);
#endif
    }
  }
#ifdef PRINT_OUTPUT
  printf("\n");
#endif
#ifdef TRIPLE
  for(i=0; i<workload and correct; i++){
    for(j=0; j<(xmax+1) and correct; j++){
      correct = float_equals(x[i * (xmax+1) + j], x_redundant2[i * (xmax+1) + j]);
#ifdef PRINT_OUTPUT
      printf("%f", x[i * (xmax+1) + j]);
#endif
    }
  }
#ifdef PRINT_OUTPUT
  printf("\n");
#endif
#endif

 #ifdef TIMING
  gettimeofday(&time_compare2, NULL);
  
  printf("Input Redundant transfer time1: %ld us\n", get_times(RedTransfer1_start, RedTransfer1_end));
  printf("Result Redundant transfer time1: %ld us\n", get_times(RedTransfer2_start, RedTransfer2_end));
  printf("Comparison time1: %ld us\n", get_times(time_compare1, time_compare2));
 #endif

  if (correct)
    printf("Redundant executions produced the same results !!!\n");
  else
    printf("ERROR: Redundant executions produced different results(2)!!!!!\n");
#endif

#ifdef TIMING
  printf("Input transfer time1: %ld us\n", get_times(Transfer1_start, Transfer1_end));
  printf("Result transfer time1: %ld us\n", get_times(Transfer2_start, Transfer2_end));
#endif

  time6 = get_time();
	//================================================================================80
	//		DEALLOCATION
	//================================================================================80

	//============================================================60
	//		X/Y INPUTS/OUTPUTS, PARAMS INPUTS
	//============================================================60

	free(y);
	cudaFree(d_y);

	free(x);
	cudaFree(d_x);

	free(params);
	cudaFree(d_params);

#ifdef REDUNDANT
	free(y_redundant);
	cudaFree(d_y_redundant);

	free(x_redundant);
	cudaFree(d_x_redundant);

	cudaFree(d_params_redundant);
#endif

#ifdef TRIPLE
	free(y_redundant2);
	cudaFree(d_y_redundant2);

	free(x_redundant2);
	cudaFree(d_x_redundant2);

	cudaFree(d_params_redundant2);
#endif

	//============================================================60
	//		TEMPORARY SOLVER VARIABLES
	//============================================================60

	cudaFree(d_com);

	cudaFree(d_err);
	cudaFree(d_scale);
	cudaFree(d_yy);

	cudaFree(d_initvalu_temp);
	cudaFree(d_finavalu_temp);

#ifdef REDUNDANT
	cudaFree(d_com_redundant);

	cudaFree(d_err_redundant);
	cudaFree(d_scale_redundant);
	cudaFree(d_yy_redundant);

	cudaFree(d_initvalu_temp_redundant);
	cudaFree(d_finavalu_temp_redundant);
  for (int i = 0; i < NUM_STREAMS; ++i)
    cudaStreamDestroy(streams[i]);
#endif

#ifdef TRIPLE
	cudaFree(d_com_redundant2);

	cudaFree(d_err_redundant2);
	cudaFree(d_scale_redundant2);
	cudaFree(d_yy_redundant2);

	cudaFree(d_initvalu_temp_redundant2);
	cudaFree(d_finavalu_temp_redundant2);
#endif

	time7= get_time();

	//================================================================================80
	//		DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%.12f s, %.12f%% : SETUP VARIABLES\n", 															(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time7-time0) * 100);
	printf("%.12f s, %.12f%% : ALLOCATE CPU MEMORY AND GPU MEMORY\n", 				(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time7-time0) * 100);
	printf("%.12f s, %.12f%% : READ DATA FROM FILES, COPY TO GPU MEMORY\n", 		(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time7-time0) * 100);
	printf("%.12f s, %.12f%% : RUN GPU KERNEL\n", 															(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time7-time0) * 100);
	printf("%.12f s, %.12f%% : COPY GPU DATA TO CPU MEMORY\n", 								(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time7-time0) * 100);
#ifdef REDUNDANT
	printf("%.12f s, %.12f%% : CHECK RESULTS WITH REDUNDANT EXECUTION\n", 																(float) (time6-time5) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);
#endif
	printf("%.12f s, %.12f%% : FREE MEMORY\n", 																(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																											(float) (time7-time0) / 1000000);

//====================================================================================================100
//		END OF FILE
//====================================================================================================100

	return 0;

}
 
 
