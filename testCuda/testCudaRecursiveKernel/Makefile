NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_60 -rdc=true
EXE=crk

test: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE) testCudaRecursiveKernel.cu

clean:
	rm -rf $(EXE)

run:
	./$(EXE)