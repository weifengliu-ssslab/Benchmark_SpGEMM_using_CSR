#compilers
CC=g++

#includes

#libs
OPENCL_LIBS = -L/usr/lib -lamdocl64
LIBS = $(OPENCL_LIBS)

all: 
	$(CC) -m64 -O3 main.cpp bhsparse.cpp bhsparse_opencl.cpp basiccl.cpp -o spgemm $(LIBS)

