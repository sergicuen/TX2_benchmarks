#include ../../common/make.config
#include ../../../Staggered_Kernel_Execution/common/make.config

ifeq ($(arg), 1)
    CCFLAGS:=
    SUFFIX:=
endif
ifeq ($(arg), 2)
    CCFLAGS:= -D TIMING
    SUFFIX:=_timing
endif
ifeq ($(arg), 3)
    CCFLAGS:= -D REDUNDANT
    SUFFIX:=_redundant
endif
ifeq ($(arg), 4)
    CCFLAGS:= -D REDUNDANT -D TIMING
    SUFFIX:=_redundant_timing
endif
ifeq ($(arg), 5)
    CCFLAGS:= -D REDUNDANT -D TRIPLE
    SUFFIX:=_redundant_triple
endif
ifeq ($(arg), 6)
    CCFLAGS:= -D REDUNDANT -D TRIPLE -D TIMING
    SUFFIX:=_redundant_triple_timing
endif
ifeq ($(arg), 7)
    CCFLAGS:= -D REDUNDANT -D SERIALIZE
    SUFFIX:=_redundant_serialize
endif
ifeq ($(arg), 8)
    CCFLAGS:= -D REDUNDANT -D SERIALIZE -D TIMING
    SUFFIX:=_redundant_serialize_timing
endif
ifeq ($(arg), 9)
    CCFLAGS:= -D REDUNDANT -D TRIPLE -D SERIALIZE
    SUFFIX:=_redundant_triple_serialize
endif
ifeq ($(arg), 10)
    CCFLAGS:= -D REDUNDANT -D TRIPLE -D TIMING -D SERIALIZE
    SUFFIX:=_redundant_triple_serialize_timing
endif

#LOCAL_CC = gcc -g -O3 -Wall
LOCAL_CC = gcc -O3 -Wall
#LOCAL_CC = gcc -O3 -Wall -D TIMING
#CUDA_DIR := /usr/local/cuda-4.0
CUDA_DIR := /usr/local/cuda
#CC := $(CUDA_DIR)/bin/nvcc
CUDA_CC := $(CUDA_DIR)/bin/nvcc -arch=sm_60 --generate-code=arch=compute_62,code=sm_62 
#CC := $(CUDA_DIR)/bin/nvcc -ccbin=g++-4.4
#EXTRAFLAGS=-L/home/salcaide/gpgpu-sim_distribution/lib/gcc-4.4.7/cuda-4000/debug -lcudart

all : mmult

clean_temp :
	rm -rf *.o 
clean :
	rm -rf *.o bin/* 

mmult : mmult.cu
	$(CUDA_CC) $(CCFLAGS) -o bin/mmult$(SUFFIX) mmult.cu 
    #$(CUDA_CC) $(CCFLAGS) -o bin/mmult$(SUFFIX) mmult.cu 

#data :
#	mkdir data
#	./gen_dataset.sh
