
debugEcho()
{
        if [ $DEBUG ]
        then
                echo $@
        fi
}

noGpuProcess()
{
        idx=0
        # make sure every line in hostfile is accessible
        while read line
        do
                debugEcho $idx $line
                if [ "$line" == "0 MiB" ]
                then
                        debugEcho good!
                else
                        debugEcho bad...
                        return 1
                fi
                idx=$(( idx + 1 ))
        done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader)
        # every card is 0 MiB, return 0
        return 0
}

setGpuFreq()
{
        #sudo nvidia-smi -ac 877,1312
        nvidia-smi -ac  958,1402
        #sudo nvidia-smi -ac  958,1425
        #sudo nvidia-smi -rac
        nvidia-smi -pm 1
}

getGpuFreq()
{
        nvidia-smi --format=csv,noheader --query-gpu=clocks.applications.memory,clocks.applications.graphics
}

# args1: hostfile to be checked
checkNet()
{
        while read -r line_ip
        do
                ping ${line_ip} -c1
        done < ${HOSTFILE}
}

checkGPU()
{
        mpirun --version
}


noSense()
{
cat <<EOF
3TunQKImbOxxwBEM

c4tzew23


/lustre/home/acct-hpc/hpccsg/software/dgx-2/openmpi3/bin/mpirun -x PATH -x LD_LIBRARY_PATH \
--mca pml ucx --mca btl ^vader,tcp,openib,uct -x UCX_TLS=rc,sm,cuda_copy,cuda_ipc \
--allow-run-as-root -np 16 -bind-to none ./run_linpack_GPU_dgx2_16


mpirun --allow-run-as-root -np 32  -hostfile ${HOSTFILE} -bind-to none -v \
       --mca btl smcuda,openib,self  --mca btl_openib_want_cuda_gdr 1 \
       --mca btl_openib_cuda_rdma_limit 30000 --mca btl_vase_verbose 100 \
       -x LD_LIBRARY_PATH ./run_linpack_GPU_dgx2_16xv100_cuda10 2>&1 | tee $RESULT_FILE




OutputDir/20190905_0527
curFiles = [ f.path for f in os.scandir(dirName) if 'ite006' in f.path ]

EOF
}