understanding CUDA to build kernels
So CPU can work like our brain on singular tasks very efficiently but cant multitask so we have GPUs which can be considered as an army of simple workers who will do multiple tasks CPU Side (Host)          GPU Side (Device)
┌─────────────────┐     ┌─────────────────┐
│   Host Memory   │────▶│  Global Memory  │ (Slow but huge)
│   (Your RAM)    │     │   (GPU VRAM)    │
└─────────────────┘     └─────────────────┘
                                │
                                ├─▶ Shared Memory (Fast, shared within block)
                                ├─▶ Constant Memory (Read-only, cached)
                                ├─▶ Texture Memory (Optimized for 2D access)
                                └─▶ Registers (Fastest, per-thread)
Memory Types:

Global Memory: Main GPU storage (like warehouse)
Shared Memory: Fast storage shared by worker teams (like team prep station)
Registers: Super fast, private to each worker (like personal tools)
Constant Memory: Read-only data (like recipe book)

Thread Indexing (Worker ID System)

__global__: Runs on GPU, called from CPU (main recipe)
__device__: Runs on GPU, called from GPU (helper function)
__host__: Runs on CPU (normal C++ function)

Allocation:
cudafloat* host_data;          // CPU memory pointer
float* device_data;        // GPU memory pointer

// Allocate on CPU (normal)
host_data = new float[1000];

// Allocate on GPU (special CUDA function)
cudaMalloc(&device_data, 1000 * sizeof(float)); 

// CPU → GPU (before processing)
cudaMemcpy(device_data, host_data, 1000 * sizeof(float), cudaMemcpyHostToDevice);

// GPU → CPU (after processing)
cudaMemcpy(host_data, device_data, 1000 * sizeof(float), cudaMemcpyDeviceToHost);

// Cleanup
delete[] host_data;
cudaFree(device_data);
                                
