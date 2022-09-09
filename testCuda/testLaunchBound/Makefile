NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_80 -rdc=true
EXE=lb.exe

test: 
	# generate kernel
	#bash generateKernel.sh
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE) testLaunchBound.cu kernel.cu

clean:
	rm -rf $(EXE)

run:
	./$(EXE)
