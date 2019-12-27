NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_60

funcptr: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o funcptr testFuncPointer.cu


init: 
	$(NVCC) $(NVCCFLAGS) -lineinfo -g -o inittest testInitStruct.cu
	./inittest

clean:
	rm -rf ./funcptr ./inittest

run:
	./funcptr