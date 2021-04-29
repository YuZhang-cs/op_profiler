# python -m torch.distributed.launch --nproc_per_node=8 --use_env pipeline.py
# mul: [8,224,5120]*[8,224,5120] -matA
# mul: [1280]*[1280]
# matmul: m=1280 n=1792 k=1280 -matA*matC
# matmul: m=64 n=224 k=224
 
#usage: python XXX.py -kernel mul --M 100 --K 100 --N 100 --P 100 --warm_up 1  --num_test 1
#usage: python XXX.py -kernel matmul --M 100 --K 100 --N 100 --P 100 --warm_up 1  --num_test 1
 
import os
import time
import torch
import argparse
 
parser = argparse.ArgumentParser(
    description="PyTorch kernel mul and matmul Benchmark"
)
 
parser.add_argument(
    "--M",
    type=int,
    default=640,
    required=False,
    help="Matrix param m",
)
parser.add_argument(
    "--K",
    type=int,
    default=1024,
    required=False,
    help="Matrix param k",
)
parser.add_argument(
    "--N",
    type=int,
    default=1880,
    required=False,
    help="Matrix param n",
)
parser.add_argument(
    "--P",
    type=int,
    default=30,
    required=False,
    help="",
)
 
parser.add_argument(
    "--warm_up", "-w", type=int, default=8, required=False, help="Num of warm up"
)
parser.add_argument(
    "--num_test", "-n", type=int, default=2000, required=False, help="Num of Test"
)
parser.add_argument(
    "--kernel",
    type=str,
    default="mul",
    required=False,
    help="the computation kernel used to run",
)
args, unknown = parser.parse_known_args()
 
M = args.M
K = args.K
N = args.N
P = args.P
 
def run(matA, matB, matC, stages):
    durations=[]
    if args.kernel == "mul":
        for stage in range(stages):
            start = time.perf_counter()
            z = matA[stage].mul(matA[stage])
            end = time.perf_counter()
            durations.append(end-start)
    elif args.kernel == "matmul":
        for stage in range(stages):
            start = time.perf_counter()
            z = matA[stage].matmul(matC)
            end = time.perf_counter()    
            durations.append(end-start)    
    return durations
        


 
def main():
    start_run = time.perf_counter()
 
    # please modify size of matrix here
    # Matrix A
    MatA = list()
    for _ in range(P):
        MatA.append(torch.randn(M, K).cuda())
    # Matrix B
    MatB = torch.randn(M, N).cuda()
    # Matrix C
    MatC = torch.randn(K, N).cuda()
    
 
    # warm up
    for i in range(args.warm_up):
        run(MatA, MatB, MatC, P)
    torch.cuda.synchronize()
    # test
    #please modify test number in the params
    durations=[]
    for i in range(args.num_test):
        start = time.perf_counter()
        # each step will run P times op, and measure time
        duration = run(MatA, MatB, MatC, P)
        torch.cuda.synchronize()
        end = time.perf_counter()
        durations.append(duration)
        #step time of op - unit: ms
        print("avg step " + str(i) +" time"+
           str((end - start) *1000*1000/P)
        )
 
    print("every op time")
    for i in range(len(durations)):
        for j in range(len(durations[0])):
            print(durations[i][j]*1000*1000)
    
    end_run = time.perf_counter()
    print("code run {}min".format((end_run - start_run) / 60))



 
if __name__ == "__main__":
    main()