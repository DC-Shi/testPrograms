NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_80 -rdc=true
EXE=crk

test: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE) testCudaRecursiveKernel.cu

clean:
	rm -rf $(EXE)

run:
	./$(EXE)
