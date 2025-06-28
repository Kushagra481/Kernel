// CUDA EXPLAINED: Your GPU Kitchen
#include <cuda_runtime.h>  // Basic GPU tools
#include <cublas_v2.h>     // Pre-made math recipes
#include <cudnn.h>         // Neural network helpers

/* 
 * KERNEL = RECIPE that thousands of GPU workers follow simultaneously
 * Each worker gets a different piece of data to work on
 */

// RECIPE 1: "Add Two Numbers" 
// Imagine 1000 workers, each adding one pair of numbers
__global__ void add_tensors(float* q, float* k, float* output, int size) {
    // "What's my worker ID?"
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // "Am I supposed to work on this piece?"
    if (idx < size) {
        output[idx] = q[idx] + k[idx];  // Worker #idx adds q[idx] + k[idx]
    }
    // Worker #0 does: output[0] = q[0] + k[0]
    // Worker #1 does: output[1] = q[1] + k[1]
    // Worker #2 does: output[2] = q[2] + k[2]
    // ... all at the SAME TIME!
}

// RECIPE 2: "Neural Network Layer"
// Like having workers do matrix multiplication in parallel
__global__ void mlp_forward(float* input, float* weights, float* bias, 
                           float* output, int batch_size, int input_dim, int output_dim) {
    
    // Each worker gets assigned a specific output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Which batch item?
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Which output feature?
    
    if (row < batch_size && col < output_dim) {
        float sum = 0.0f;
        
        // This worker computes ONE output value
        for (int k = 0; k < input_dim; k++) {
            sum += input[row * input_dim + k] * weights[k * output_dim + col];
        }
        
        // Apply ReLU activation (like adding seasoning)
        output[row * output_dim + col] = fmaxf(0.0f, sum + bias[col]);
    }
}

// RECIPE 3: "Apply Seasoning Based on Taste Score"
__global__ void apply_attention(float* scores, float* v, float* output, 
                               int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_pos = idx / dim;      // Which sequence position?
    int feat_pos = idx % dim;     // Which feature?
    
    if (seq_pos < seq_len && feat_pos < dim) {
        // Use the attention score to weight the value
        output[idx] = scores[seq_pos] * v[idx];
    }
}

/*
 * THE MAIN KITCHEN CLASS
 * This manages all your GPU memory and coordinates the workers
 */
class MinimalAttention {
private:
    cublasHandle_t cublas_handle;  // Your math assistant
    
    // GPU MEMORY SPACES (like prep stations in kitchen)
    float *d_q, *d_k, *d_v;           // Input ingredients (on GPU)
    float *d_qk_sum, *d_scores, *d_output; // Work surfaces (on GPU)
    float *d_mlp_weights, *d_mlp_bias;     // Recipe parameters (on GPU)
    
    int seq_len, dim, mlp_hidden;     // Kitchen size parameters

public:
    // CONSTRUCTOR: Set up your kitchen
    MinimalAttention(int seq_len, int dim, int mlp_hidden) 
        : seq_len(seq_len), dim(dim), mlp_hidden(mlp_hidden) {
        
        // Get your math assistant ready
        cublasCreate(&cublas_handle);
        
        // ALLOCATE GPU MEMORY (like buying prep stations)
        // Think: "I need space for X numbers on the GPU"
        cudaMalloc(&d_q, seq_len * dim * sizeof(float));      // Space for q
        cudaMalloc(&d_k, seq_len * dim * sizeof(float));      // Space for k  
        cudaMalloc(&d_v, seq_len * dim * sizeof(float));      // Space for v
        cudaMalloc(&d_qk_sum, seq_len * dim * sizeof(float)); // Work surface
        cudaMalloc(&d_scores, seq_len * sizeof(float));       // Scores
        cudaMalloc(&d_output, seq_len * dim * sizeof(float)); // Final dish
        
        // Recipe storage (MLP weights)
        cudaMalloc(&d_mlp_weights, dim * sizeof(float));
        cudaMalloc(&d_mlp_bias, sizeof(float));
    }
    
    // THE MAIN COOKING PROCESS
    void forward(float* h_q, float* h_k, float* h_v, float* h_output) {
        
        // STEP 1: Move ingredients from CPU kitchen to GPU kitchen
        // (Like moving ingredients from pantry to prep stations)
        cudaMemcpy(d_q, h_q, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v, seq_len * dim * sizeof(float), cudaMemcpyHostToDevice);
        
        // STEP 2: ADD Q + K (Launch Recipe #1)
        // Grid/Block = "How to organize your workers"
        dim3 block(256);  // 256 workers per team
        dim3 grid((seq_len * dim + block.x - 1) / block.x);  // How many teams needed?
        
        // "Hey workers! Follow recipe #1!"
        add_tensors<<<grid, block>>>(d_q, d_k, d_qk_sum, seq_len * dim);
        
        // STEP 3: MLP PROCESSING (Launch Recipe #2)
        dim3 mlp_block(16, 16);  // 16x16 = 256 workers per team, arranged in 2D
        dim3 mlp_grid((1 + mlp_block.x - 1) / mlp_block.x,           // X dimension
                      (seq_len + mlp_block.y - 1) / mlp_block.y);    // Y dimension
        
        // "Hey workers! Follow recipe #2!"
        mlp_forward<<<mlp_grid, mlp_block>>>(d_qk_sum, d_mlp_weights, d_mlp_bias,
                                           d_scores, seq_len, dim, 1);
        
        // STEP 4: APPLY ATTENTION (Launch Recipe #3)
        // "Hey workers! Follow recipe #3!"
        apply_attention<<<grid, block>>>(d_scores, d_v, d_output, seq_len, dim);
        
        // STEP 5: Bring final dish back to CPU kitchen
        cudaMemcpy(h_output, d_output, seq_len * dim * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        // Wait for all workers to finish
        cudaDeviceSynchronize();
    }
    
    // DESTRUCTOR: Clean up the kitchen
    ~MinimalAttention() {
        // Free all GPU memory (return prep stations)
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_qk_sum); cudaFree(d_scores); cudaFree(d_output);
        cudaFree(d_mlp_weights); cudaFree(d_mlp_bias);
        
        // Dismiss your math assistant
        cublasDestroy(cublas_handle);
    }
};

// HOW TO USE YOUR GPU KITCHEN
int main() {
    int seq_len = 128, dim = 512;  // Kitchen size
    
    // Regular CPU memory (your main kitchen)
    float *q = new float[seq_len * dim];       // Ingredient A
    float *k = new float[seq_len * dim];       // Ingredient B  
    float *v = new float[seq_len * dim];       // Ingredient C
    float *output = new float[seq_len * dim];  // Final dish
    
    // Fill with your data here...
    
    // Set up GPU kitchen and cook!
    MinimalAttention attention(seq_len, dim, 256);
    attention.forward(q, k, v, output);
    
    // Your result is now in 'output'!
    
    // Clean up CPU memory
    delete[] q; delete[] k; delete[] v; delete[] output;
    return 0;
}
