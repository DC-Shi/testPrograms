PROJECT_NAME = test_calc

# NVCC is path to nvcc. Here it is assumed /usr/local/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = gcc

CUDAPATH = /usr/local/cuda

BUILD_DIR = build
OBJ_DIR = ./obj
SRC_DIR = ./src
INC_DIR = ./inc
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

CC_FLAGS = -c -I$(CUDAPATH)/include -I$(INC_DIR)
CFLAGS = -I$(CUDAPATH)/include -I.
NVCC_FLAGS = -I$(CUDAPATH)/include -I$(INC_DIR) -gencode arch=compute_60,code=sm_60 -O0 -lineinfo

# this is mac only. only linux system, remove -Xlinker and add in -lGL -lGLUT?
LFLAGS = -L$(CUDAPATH)/lib64 -L. -lcudart -lm 

# Target executable name:
EXE = build/$(PROJECT_NAME)

# Object files:
CPU_OBJS = $(OBJ_DIR)/cpu_side.o $(OBJ_DIR)/main.o
CUDA_O = $(OBJ_DIR)/bugcuda.o
CUDAOBJS =  $(OBJ_DIR)/kernel.o $(OBJ_DIR)/bug.o
OBJS = $(CUDA_O) $(CPU_OBJS) $(CUDAOBJS)

.PHONY: clean


# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	g++ -o $@ $(OBJS) -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm -lcudadevrt -I$(INC_DIR)

# Compile main .cpp file to object files:
#$(OBJ_DIR)/%.o : %.cpp
#	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA obj files to one object file:
$(OBJ_DIR)/bugcuda.o : $(CUDAOBJS)
	$(NVCC) $(NVCC_FLAGS) -dlink $(CUDAOBJS) -o $@ $(NVCC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@ $(NVCC_LIBS)

clean:
	rm -rf $(OBJ_DIR)/*.o $(EXE)

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)