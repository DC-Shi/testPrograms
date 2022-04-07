NVCC=nvcc
NVCCFLAGS=-I../inc -arch=sm_80
EXECUTABLE=exe.exe

$(EXECUTABLE) : 
	$(NVCC) $(NVCCFLAGS) -lineinfo -O0 -rdc=true -g -G -o $@ main.cu -lcudadevrt

run : $(EXECUTABLE)
	CUDA_VISIBLE_DEVICES=3 ./$(EXECUTABLE)

clean:
	rm -rf ./*.o $(EXECUTABLE)
