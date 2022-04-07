NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_80 -O0
EXE=wildPointer.exe

test: 
	$(NVCC) $(NVCCFLAGS) -g -o $(EXE) testWildPointer.cu

clean:
	rm -rf $(EXE)

run:
	./$(EXE)
