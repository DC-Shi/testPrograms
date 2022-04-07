NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_80 -gencode=arch=compute_75,code=sm_75
EXE1=scheduling.exe
EXE2=sleep.exe
EXE3=loop.exe


loop: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE3) loop.cu

test: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE1) testScheduling.cu

sleep: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -DSLEEP -g -o $(EXE2) testScheduling.cu

clean:
	rm -rf $(EXE1) $(EXE2) $(EXE3)

run:
	./$(EXE1)
