#include "lab1.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector_types.h>
#include <cuComplex.h>
#include <png.h>
#include <string.h>

#include <fstream>
#include <unistd.h>

#define CUDA_CHECK_RETURN(value) {											\
		cudaError_t _m_cudaStat = value;										\
		if (_m_cudaStat != cudaSuccess) {										\
			fprintf(stderr, "Error %s at line %d in file %s\n",					\
					cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
					exit(1);															\
		} }

using namespace std;

#define TYPEFLOAT

#ifdef TYPEFLOAT
#define TYPE float
#define cTYPE cuFloatComplex
#define cMakecuComplex(re,i) make_cuFloatComplex(re,i)
#endif

#define moveX (0)
#define moveY (0)
#define MAXITERATIONS  (256)
static const int DIMX = 640;
static const int DIMY = 480;
static const int BLOCKX = 32;
static const int BLOCKY = 32;
static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

// Julia function //
/** a useful function to compute the number of threads **/
int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }


cTYPE c0;

__device__ cTYPE juliaFunctor(cTYPE p,cTYPE c){
	return cuCaddf(cuCmulf(p,p),c);
}

__device__ int evolveComplexPoint(cTYPE p,cTYPE c){
	int it =1;
	while(it <= MAXITERATIONS && cuCabsf(p) <=4){
		p=juliaFunctor(p,c);
		it++;
	}
	return it;
}

__device__ cTYPE convertToComplex(int x, int y,float zoom){
	//	TYPE factor = sqrt((DIMX/2.0))* sqrt((DIMX/2.0)) + (DIMY/2)*(DIMY/2) ;
	TYPE jx = 1.5 * (x - DIMX / 2) / (0.5 * zoom * DIMX) + moveX;
	TYPE jy = (y - DIMY / 2) / (0.5 * zoom * DIMY) + moveY;
	return cMakecuComplex(jx,jy);
}


__global__ void computeJulia(uint8_t* data,cTYPE c,float zoom){
	int i =  blockIdx.x * blockDim.x + threadIdx.x;
	int j =  blockIdx.y * blockDim.y + threadIdx.y;

	if(i<DIMX && j<DIMY){
		cTYPE p = convertToComplex(i,j,zoom);
		data[i*DIMY+j] = evolveComplexPoint(p,c);
	}

}
// end Julia function //


struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) {
    // Julia function // 
    size_t dataSize = sizeof(uint8_t)*DIMX*DIMY;
    uint8_t *hdata;
	CUDA_CHECK_RETURN(cudaMallocHost(&hdata, dataSize));
    uint8_t *ddata;
	CUDA_CHECK_RETURN(cudaMalloc(&ddata, dataSize));

	dim3 bs(BLOCKX, BLOCKY);
	dim3 gs(divup(DIMX, bs.x), divup(DIMY, bs.y));
	float incre = 0.00000015;
	float inci = -0.00045;
	float startre = -0.75;
	float starti = 0.09;
	float zoom = 1 + (impl->t)*0.005;
    
    c0 = cMakecuComplex(startre + (impl->t)*incre, starti + (impl->t)*inci);

	computeJulia<<<gs,bs>>>(ddata,c0,zoom);
	CUDA_CHECK_RETURN(cudaMemcpy(hdata, ddata, dataSize, cudaMemcpyDeviceToHost));
    cudaFree(ddata);
     
    CUDA_CHECK_RETURN(cudaMemcpy(yuv, hdata, dataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemset(yuv+DIMX*DIMY, 128, DIMX*DIMY/2));
	// end Julia function //

    cudaFreeHost(hdata);
	++(impl->t);
}
