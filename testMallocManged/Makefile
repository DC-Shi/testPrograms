
NVCC = nvcc
NVCC_FLAGS = -arch=compute_70 -code=sm_70 -g

.PHONY: clean

all: initTryMultiGpu

add:
	$(NVCC) $(NVCC_FLAGS) add_grid.cu


initTryMultiGpu:
	$(NVCC) $(NVCC_FLAGS) initTryMultiGpu.cu

simpleinit:
	$(NVCC) $(NVCC_FLAGS) simpleinit.cu
oldinit:
	$(NVCC) $(NVCC_FLAGS) oldinit.cu

clean:
	rm -rf a.out

run:
	./a.out