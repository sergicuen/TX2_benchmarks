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


NVCC := nvcc
CUDA_DIR := /usr/local/cuda
NVCC := $(CUDA_DIR)/bin/nvcc
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
INCLUDE := $(CUDA_DIR)/include

myocyte.out:		main.o
	$(NVCC)	main.o \
				--compiler-options="$(CCFLAGS)"\
				-I${INCLUDE} \
				-L${CUDA_LIB_DIR} \
				-lm -lcuda -lcudart \
                                -o bin/myocyte.out$(SUFFIX)
				#--ptxas-options=-v \


# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	$(NVCC)	main.cu -c --compiler-options="$(CCFLAGS)" -O3 -g #--ptxas-options=-v \

# delete all object files
clean_temp:
	rm -f *.o output.txt
clean:
	rm -f *.o bin/* output.txt
