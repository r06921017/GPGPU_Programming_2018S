#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>


#include <iostream>
#include "SyncedMemory.h"
using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }


struct char_encode{
    __device__  int operator()(char c) const{
        return (c =='\n')? 0 : 1;
    }
};

__global__ void count_pos_kernel(const char* text, int* pos, int text_size){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx <= text_size - 1){
        int count = 0;

        for (int i = 0; i <= idx; i++){
            if (text[idx-i] != '\n'){
                count ++;
            }

            else{
                break;
            }
        }

        pos[idx] = count;
    }
    return;
}


void CountPosition1(const char *text, int *pos, int text_size)
{
    auto ptr_text = thrust::device_pointer_cast(text);
    auto ptr_pos = thrust::device_pointer_cast(pos);

    auto char_enc = thrust::make_transform_iterator(ptr_text, char_encode());
    thrust::inclusive_scan_by_key(char_enc, char_enc+text_size, char_enc, ptr_pos);
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    count_pos_kernel <<<CeilDiv(text_size, 512), 512>>>(text, pos, text_size);
}
