#!/bin/bash

OUTPUTDIR1=Profiler_sessions
OUTPUTDIR2=Results_profiler
#NUMBER OF DEVICE EXECUTIONS PER EACH TEST
NUMEX=100

mkdir -p ${OUTPUTDIR1}
mkdir -p ${OUTPUTDIR2}

echo Starting MYOCYTE redundant executions
for it in `seq 1 ${NUMEX}`;
	do
	CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace	-o ${OUTPUTDIR1}/Test_${it}_redundant.nvprof --csv ./bin/myocyte_redundant.out 100 1 1 &> ${OUTPUTDIR2}/results_redundant_${it}

done
echo Ended redundant executions

echo Starting MYOCYTE executions
for it in `seq 1 ${NUMEX}`;
	do
	CUDA_VISIBLE_DEVICES=1 /usr/local/cuda/bin/nvprof --print-gpu-trace --print-api-trace	-o ${OUTPUTDIR1}/Test_${it}.nvprof --csv ./bin/myocyte.out 100 1 1  &> ${OUTPUTDIR2}/results_${it}

done
echo Ended executions
