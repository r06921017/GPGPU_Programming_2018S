#include "lab3.h"
#include <cstdio>
#include "Timer.h"
#include <iostream>
using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
    	const float *background,
	    const float *target,
	    const float *mask,
    	float *output,
	    const int wb, const int hb, const int wt, const int ht,
    	const int oy, const int ox)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}


__global__ void CalculateFixed(
        const float* background, 
        const float* target, 
        const float* mask,
        float* fixed,
        const int wb, const int hb, const int wt, const int ht,
        const int oy, const int ox)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;

    const int curt = wt*yt + xt;

    if (yt < ht && xt < wt && mask[curt] > 127.0f){
        int north_t = (yt == 0) ? curt : (curt-wt);
        int west_t  = (xt == 0) ? curt : (curt-1);
        int south_t = (yt == (ht-1)) ? curt : (curt+wt);
        int east_t  = (xt == (wt-1)) ? curt : (curt+1);

        fixed[curt*3+0] = 4.0f*target[curt*3+0] - (target[north_t*3+0] + target[west_t*3+0] + target[south_t*3+0] + target[east_t*3+0]);
        fixed[curt*3+1] = 4.0f*target[curt*3+1] - (target[north_t*3+1] + target[west_t*3+1] + target[south_t*3+1] + target[east_t*3+1]);
        fixed[curt*3+2] = 4.0f*target[curt*3+2] - (target[north_t*3+2] + target[west_t*3+2] + target[south_t*3+2] + target[east_t*3+2]);

        // Calculate background replaced by the target position
        const int xb = ox + xt;
        const int yb = oy + yt;
        const int curb = wb*yb + xb;

        int north_b = (yb == 0) ? curb : (curb-wb);
        int west_b  = (xb == 0) ? curb : (curb-1);
        int south_b = (yb == (hb-1)) ? curb : (curb+wb);
        int east_b  = (yb == (wb-1)) ? curb : (curb+1);

        // Check target is boundary or masked
        if (yt==0 || mask[north_t] <= 127.0f){
            fixed[curt*3+0] += background[north_b*3+0];
            fixed[curt*3+1] += background[north_b*3+1];
            fixed[curt*3+2] += background[north_b*3+2];
        }

        if (xt==0 || mask[west_t] <= 127.0f){
            fixed[curt*3+0] += background[west_b*3+0];
            fixed[curt*3+1] += background[west_b*3+1];
            fixed[curt*3+2] += background[west_b*3+2];
        }

        if (yt==(ht-1) || mask[south_t] <= 127.0f){
            fixed[curt*3+0] += background[south_b*3+0];
            fixed[curt*3+1] += background[south_b*3+1];
            fixed[curt*3+2] += background[south_b*3+2];
        }
        
        if (xt==(wt-1) || mask[east_t] <= 127.0f){
            fixed[curt*3+0] += background[east_b*3+0];
            fixed[curt*3+1] += background[east_b*3+1];
            fixed[curt*3+2] += background[east_b*3+2];
        }
    }
}


__global__ void PoissonImageCloningIteration(
        const float* fixed, 
        const float* mask,
        const float* buf1,
        float* buf2, 
        const int wt, const int ht)
{
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int curt = wt*yt + xt;
    
    if (yt < ht && xt < wt && mask[curt] > 127.0f){
        int north_t = (yt == 0) ? curt : (curt-wt);
        int west_t  = (xt == 0) ? curt : (curt-1);
        int south_t = (yt == (ht-1)) ? curt : (curt+wt);
        int east_t  = (xt == (wt-1)) ? curt : (curt+1);

        buf2[curt*3+0] = fixed[curt*3+0];
        buf2[curt*3+1] = fixed[curt*3+1];
        buf2[curt*3+2] = fixed[curt*3+2];

        if (yt!=0 && mask[north_t] > 127.0f){
            buf2[curt*3+0] += buf1[north_t*3+0];
            buf2[curt*3+1] += buf1[north_t*3+1];
            buf2[curt*3+2] += buf1[north_t*3+2];
        }

        if (xt!=0 && mask[west_t] > 127.0f){
            buf2[curt*3+0] += buf1[west_t*3+0];
            buf2[curt*3+1] += buf1[west_t*3+1];
            buf2[curt*3+2] += buf1[west_t*3+2];
        }

        if (yt!=(ht-1) && mask[south_t] > 127.0f){
            buf2[curt*3+0] += buf1[south_t*3+0];
            buf2[curt*3+1] += buf1[south_t*3+1];
            buf2[curt*3+2] += buf1[south_t*3+2];
        }
        
        if (xt!=(wt-1) && mask[east_t] > 127.0f){
            buf2[curt*3+0] += buf1[east_t*3+0];
            buf2[curt*3+1] += buf1[east_t*3+1];
            buf2[curt*3+2] += buf1[east_t*3+2];
        }


        buf2[curt*3+0] *= 0.25f;
        buf2[curt*3+1] *= 0.25f;
        buf2[curt*3+2] *= 0.25f;
    }
}


// Downsampling the fixed/mask
__global__ void AvgPooling(
        const float *mask,
        const float *fixed,
        const float *buf1,
        float *mask_scaled,
        float *fixed_scaled, 
        float *buf1_scaled,
        const int wt, const int ht,
        const int scale)
{
    const int wt_scaled = wt / scale;
    const int ht_scaled = ht / scale;

    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;

    const int curt = wt_scaled*yt + xt;
    float count=0.0f;

    if (yt >= 0 && xt >= 0 && yt < ht_scaled && xt < wt_scaled){
        for (int c = 0; c < 3; c++){
            count = 0.0f;

            for (int i = xt*scale; i < (xt+1)*scale && i<wt; i++){
                for (int j = yt*scale; j < (yt+1)*scale && j < ht; j++){
                    fixed_scaled[3*curt+c] += fixed[3*(j*wt+i)+c];
                    buf1_scaled[3*curt+c] += buf1[3*(j*wt+i)+c];
                    count += 1.0;
                }
            }

            fixed_scaled[3*curt+c] /= count;
            buf1_scaled[3*curt+c] /= count;
        }

        count = 0.0f;
        for (int i = xt*scale; i < (xt+1)*scale && i<wt; i++){
            for (int j = yt*scale; j < (yt+1)*scale && j < ht; j++){
                mask_scaled[curt] += mask[j*wt+i];
                count += 1.0;
            }
        }
        mask_scaled[curt] /= count;
    }
}


// Upsampling the fixed/mask
__global__ void UpSampling(
        float *buf1,
        const float *buf1_scaled,
        const int wt, const int ht,
        int scale)
{
    const int wt_scaled = wt / scale;
    const int ht_scaled = ht / scale;

    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int curt = wt_scaled * yt + xt;

    if (yt >= 0 && xt >= 0 && yt < ht_scaled && xt < wt_scaled){
        for (int i = xt*scale; i < (xt+1)*scale; i++){
            for (int j = yt*scale; j < (yt+1)*scale; j++){
                buf1[3*(j*wt+i)+0] = buf1_scaled[3*curt+0];
                buf1[3*(j*wt+i)+1] = buf1_scaled[3*curt+1];
                buf1[3*(j*wt+i)+2] = buf1_scaled[3*curt+2];
            }
        }
    }
}


void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
    // set up
    float *fixed, *buf1, *buf2, *fixed_scaled, *mask_scaled, *buf1_scaled;
    cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1,  3*wt*ht*sizeof(float));
    cudaMalloc(&buf2,  3*wt*ht*sizeof(float));
    
    cudaMalloc(&mask_scaled,  wt*ht*sizeof(float));
    cudaMalloc(&fixed_scaled, 3*wt*ht*sizeof(float));
    cudaMalloc(&buf1_scaled,  3*wt*ht*sizeof(float));

    // Start to count the time
	Timer timer_count_position;
	timer_count_position.Start();
    
    // initialize the iteration
    CalculateFixed <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
        background, target, mask, fixed, wb, hb, wt, ht, oy, ox 
    );

    // Copy the target image to buf1
    cudaMemcpy(buf1, target, 3*wt*ht*sizeof(float), cudaMemcpyDeviceToDevice);

    
    int iter_num = 1000;
    int scale = 16;
    
    // iterate at 1/16 scale
    AvgPooling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                mask, fixed, buf1, mask_scaled, fixed_scaled, buf1_scaled, wt, ht, scale         
    );
    for (int i =0; i < iter_num; ++i){
        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/scale, ht/scale      
        );

        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/scale, ht/scale      
        );
    }
    UpSampling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            buf1, buf1_scaled, wt, ht, scale 
    );

    // iterate at 1/8 scale
    scale = 8;
    AvgPooling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                mask, fixed, buf1, mask_scaled, fixed_scaled, buf1_scaled, wt, ht, scale         
    );

    for (int i =0; i < iter_num; ++i){
        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/scale, ht/scale      
        );

        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/scale, ht/scale      
        );
    }

    UpSampling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            buf1, buf1_scaled, wt, ht, scale 
    );
    

    // iterate at 1/4 scale
    scale = 4;
    AvgPooling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                mask, fixed, buf1, mask_scaled, fixed_scaled, buf1_scaled, wt, ht, scale         
    );

    for (int i =0; i < iter_num; ++i){
        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/scale, ht/scale      
        );

        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/scale, ht/scale      
        );
    }

    UpSampling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            buf1, buf1_scaled, wt, ht, scale 
    );
    
    // iterate at 1/2 scale
    scale = 2;
    AvgPooling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                mask, fixed, buf1, mask_scaled, fixed_scaled, buf1_scaled, wt, ht, scale         
    );


    for (int i =0; i < iter_num; ++i){
        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf1_scaled, buf2, wt/scale, ht/scale      
        );

        PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
                fixed_scaled, mask_scaled, buf2, buf1_scaled, wt/scale, ht/scale      
        );
    }

    UpSampling <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            buf1, buf1_scaled, wt, ht, scale 
    );

    // iterate at original scale
    for (int i = 0; i < 6000; i++){
       PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
               fixed, mask, buf1, buf2, wt, ht 
       );

       PoissonImageCloningIteration <<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
               fixed, mask, buf2, buf1, wt, ht 
       );      
    
    }
    
    timer_count_position.Pause();
    printf_timer(timer_count_position); // Show the time

    // copy to output
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);

    // free the memory
    cudaFree(fixed);
    cudaFree(buf1);
    cudaFree(buf2);
    cudaFree(fixed_scaled);
    cudaFree(buf1_scaled);
    cudaFree(mask_scaled);
}
