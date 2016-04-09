#compilers
CC=nvcc

#includes
CUDA_INSTALL_PATH = /usr/local/cuda-7.5
INCLUDES = -I$(CUDA_INSTALL_PATH)/include -I$(CUDA_INSTALL_PATH)/samples/common/inc

#libs
CUDA_LIBS = -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
LIBS = $(CUDA_LIBS)

#-gencode=arch=compute_35,code=sm_35
#-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52

all: 
	nvcc -O3 -m64 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52 main.cu -o spgemm $(INCLUDES) $(LIBS)

