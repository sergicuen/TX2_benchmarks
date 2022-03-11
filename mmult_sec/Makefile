

NVCC=/usr/local/cuda/bin/nvcc

CCFLAGS= -O0 -m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
BINDIR=bin
SRCS=$(wildcard src/*.cu)

#for each file in src compile twice
all: $(SRCS:.cu=)

%: %.cu
	${NVCC} ${CCFLAGS} $< -o ${BINDIR}/$(basename $(notdir $@))

.PHONY: clean

clean:
	rm -f ${BINDIR}/*
