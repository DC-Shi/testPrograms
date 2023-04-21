NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_90 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_70,code=sm_70
EXE=mgpu.exe


mgpu: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o $(EXE) mgpu.cu


clean:
	rm -rf $(EXE) 

run:
	./$(EXE)
