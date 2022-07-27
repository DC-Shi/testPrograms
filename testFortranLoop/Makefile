FCFLAGS= -gpu=cc80,lineinfo -Minfo=accel
FCACCFLAGS = -acc
FC     = pgfortran
SRC    = loopExpand.F90
NSYS   = nsys profile --force-overwrite=true
NCU    = ncu --target-processes application-only \
    --replay-mode kernel --kernel-name-base function --kill no \
    --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart \
    --section MemoryWorkloadAnalysis_Tables --section Nvlink --section Nvlink_Tables \
    --section Nvlink_Topology --section Occupancy --section SchedulerStats \
    --section SourceCounters --section SpeedOfLight \
    --section SpeedOfLight_HierarchicalDoubleRooflineChart \
    --section SpeedOfLight_HierarchicalHalfRooflineChart \
    --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section SpeedOfLight_HierarchicalTensorRooflineChart \
    --section SpeedOfLight_RooflineChart --section WarpStateStats \
    --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 \
    --nvtx --profile-from-start 1 --cache-control all --clock-control base \
    --apply-rules yes --import-source no --check-exit-code yes


all: CUDA_IJK CUDA_KJI OPENACC_IJK OPENACC_KJI

cpu:
	$(FC) -Minfo=acces -o cpu_ijk.exe -DORDERIJK $(SRC)

OPENACC_IJK:
	$(FC) $(FCACCFLAGS) $(FCFLAGS) -o openacc_ijk.exe -DORDERIJK $(SRC)

OPENACC_KJI:
	$(FC) $(FCACCFLAGS) $(FCFLAGS) -o openacc_kji.exe -DORDERKJI $(SRC)

CUDA_IJK:
	$(FC) -cuda $(FCFLAGS) -o cuda_ijk.exe -DORDERIJK $(SRC)

CUDA_KJI:
	$(FC) -cuda $(FCFLAGS) -o cuda_kji.exe -DORDERKJI $(SRC)

profile: all
	$(NSYS) -o profile_acc_ijk ./openacc_ijk.exe
	$(NSYS) -o profile_acc_kji ./openacc_kji.exe
	$(NSYS) -o profile_cuda_ijk ./cuda_ijk.exe
	$(NSYS) -o profile_cuda_kji ./cuda_kji.exe

kernel_profile: profile
	$(NCU) -o kernel_acc_ijk ./openacc_ijk.exe
	$(NCU) -o kernel_acc_kji ./openacc_kji.exe
	$(NCU) -o kernel_cuda_ijk ./cuda_ijk.exe
	$(NCU) -o kernel_cuda_kji ./cuda_kji.exe

clean:
	rm *.exe
