NVCC = nvcc
NVCC_FLAGS = -arch=compute_80 -code=sm_80 -lineinfo -g

.PHONY: clean

all: naive optimise

naive:
	$(NVCC) $(NVCC_FLAGS) -DNAIVE -o naive.exe transpose.cu

optimise:
	$(NVCC) $(NVCC_FLAGS) -DOPTIMISE32 -o optimise.exe transpose.cu


clean:
	rm -rf a.out naive.exe optimise.exe
