"""Compile CUDA, verify GPU results, and retain a counter across disposable VMs."""
import json
import os
from pathlib import Path
import subprocess

output = Path(os.environ.get("COLAB_OUTPUT_DIR", "outputs"))
output.mkdir(parents=True, exist_ok=True)
source = Path("vector_add.cu")
source.write_text(r'''
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s\n",cudaGetErrorString(e)); exit(1); }} while(0)
__global__ void twice(int* values) { int i=threadIdx.x; values[i]=i*2; }
int main() {
    int *device, host[256]; cudaDeviceProp properties;
    CUDA(cudaGetDeviceProperties(&properties,0));
    CUDA(cudaMalloc(&device,sizeof(host)));
    twice<<<1,256>>>(device); CUDA(cudaGetLastError());
    CUDA(cudaMemcpy(host,device,sizeof(host),cudaMemcpyDeviceToHost));
    for(int i=0;i<256;i++) if(host[i]!=i*2) return 2;
    CUDA(cudaFree(device));
    printf("PASS: 256 GPU results verified on %s\n",properties.name);
}
''')
subprocess.run(["/usr/local/cuda/bin/nvcc", str(source), "-o", "vector_add"], check=True)
result = subprocess.check_output(["./vector_add"], text=True)
print(result, end="")
(output / "cuda-result.txt").write_text(result)
counter = output / "resume-counter.json"
previous = json.loads(counter.read_text())["runs"] if counter.exists() else 0
temporary = counter.with_suffix(".tmp")
temporary.write_text(json.dumps({"runs": previous + 1, "previous_runs_restored": previous}) + "\n")
temporary.replace(counter)
print(f"Persistent run count: {previous + 1}; restored previous count: {previous}")
