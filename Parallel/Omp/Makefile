
CXX = g++
CFLAGS = `pkg-config --cflags opencv` -lboost_filesystem -lboost_system
LIBS = `pkg-config --libs opencv` -lboost_filesystem -lboost_system

.PHONY: all clean

all: main

main: main.cpp
	$(CXX) -fopenmp $(CFLAGS) -std=c++11 main.cpp -o $@ $(LIBS) $(CFLAGS)

clean:
	rm -f main