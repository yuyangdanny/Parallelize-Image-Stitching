NVCC = nvcc
CFLAGS = -std=c++11 -O3 `pkg-config --cflags opencv`  -lboost_filesystem -lboost_system -Xcompiler -fopenmp
LIBS = `pkg-config --libs opencv`  -lboost_filesystem -lboost_system 
INCLUDES = -I/usr/local/cuda/include
LFLAGS = -L/usr/local/cuda/lib64 -lcudart -lgomp

.PHONY: all clean

all: main

main: main.cu
	$(NVCC) $(CFLAGS) $(INCLUDES) main.cu -o $@ $(LIBS) $(LFLAGS)

clean:
	rm -f main
