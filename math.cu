#include "math.cuh"

/////////////////////////////////////////////////////////
/// CPU section

void Math::CPUadd(Matrix& a, Matrix& b, Matrix& result) {
	for (int i = 0; i < a.GetSize(); i++) {
		for (int j = 0; j < a.GetSize(); j++) {
			int indeks = i * a.GetSize() + j;
			result.at(indeks) = a.at(indeks) + b.at(indeks);
		}
	}
}

void Math::CPUsubtract(Matrix& a, Matrix& b, Matrix& result) {
	for (int i = 0; i < a.GetSize(); i++) {
		for (int j = 0; j < a.GetSize(); j++) {
			int indeks = i * a.GetSize() + j;
			result.at(indeks) = a.at(indeks) - b.at(indeks);
		}
	}
}

void Math::CPUtranspose(Matrix& a, Matrix &result) {
	for (int i = 0; i < a.GetSize(); i++) {
		for (int j = 0; j < a.GetSize(); j++) {
			int indeks = i * a.GetSize() + j;
			int swapindeks = j * a.GetSize() + i;
			result.at(indeks) = a.at(swapindeks);
		}
	}
}

void Math::CPUmultiply(Matrix& a, Matrix& b, Matrix& result) {
	for (int i = 0; i < a.GetSize(); i++) {
		for (int j = 0; j < a.GetSize(); j++) {
			int indeks = i * a.GetSize() + j;
			for (int k = 0; k < a.GetSize(); k++) {
				int aindeks = i * a.GetSize() + k;
				int bindeks = k * a.GetSize() + j;
				result.at(indeks) += a.at(aindeks) * b.at(bindeks);

			}
		}
	}
}

void Math::CPUscalarmul(Matrix& a, float scalar) {
	for (int i = 0; i < a.GetSize(); i++) {
		for (int j = 0; j < a.GetSize(); j++) {
			int indeks = i * a.GetSize() + j;
			a.at(indeks) = a.at(indeks) * scalar;
		}
	}
}


/////////////////////////////////////////////////////////////
/// GPU section
__global__ void dGPUadd(float* a, float* b, float* result, int maxindex) {

	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < maxindex; i += gridDim.x * blockDim.x) {
		result[i] = a[i] + b[i];
	}

}
void Math::GPUadd(Matrix& a, Matrix& b, Matrix& result) {

	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	int threads = 1024;
	
	int maxindex = (a.GetSize()* a.GetSize());
	dGPUadd<<< 8 * numSMs, threads>>>(a.GetDevPointer(), b.GetDevPointer(), result.GetDevPointer(), maxindex);
	CUDA_CALL(cudaGetLastError());
}
__global__ void dGPUsubtract(float* a, float* b, float* result, int maxindex) {


	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < maxindex; i += gridDim.x * blockDim.x) {
		result[i] = a[i] - b[i];
	}

}
void Math::GPUsubtract(Matrix& a, Matrix& b, Matrix& result) {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	int threads = 1024;

	int maxindex = (a.GetSize() * a.GetSize());
	dGPUsubtract << < 8 * numSMs, threads >> > (a.GetDevPointer(), b.GetDevPointer(), result.GetDevPointer(), maxindex);
	CUDA_CALL(cudaGetLastError());
}
__global__ void dGPUtranspose(float* a, float* result) {

	// shared memory allocation ensuring that there are no bank conflicts
	__shared__ float subblock[TILE_DIM][TILE_DIM + 1];

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	// coalesced global memory access: 32 threads in x axis read contigous data 
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		subblock[threadIdx.y + i][threadIdx.x] = a[(y + i) * width + x];
	}
	
	__syncthreads();

	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;

	// coalesced global memory write: 32 threads in x axis write contigous data
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		result[(y + i) * width + x] = subblock[threadIdx.x][threadIdx.y + i];
	}
}

void Math::GPUtranspose(Matrix& a, Matrix& result) {

	dim3 blocknum(a.GetSize() / TILE_DIM, a.GetSize() / TILE_DIM, 1);
	dim3 threadnum(TILE_DIM, BLOCK_ROWS);

	dGPUtranspose << <blocknum, threadnum >> > (a.GetDevPointer(), result.GetDevPointer());
}
__global__ void dGPUmultiply(float* a, float* b, float* result, int width) {
	int indeks = blockIdx.x * blockDim.x + threadIdx.x;
	int col = indeks % width;
	int row = indeks / width;
	if (indeks < width * width) {
		float value = 0;
		for (int i = 0; i < width; i++) {
			value += a[row * width + i] * b[i * width + col];
		}
		result[row * width + col] = value;
	}
}

void Math::GPUmultiply(Matrix& a, Matrix& b, Matrix& result) {
	int threads = 512;
	int blocks = (a.GetSize()* a.GetSize()) / threads + 1;
	int width = a.GetSize();
	dGPUmultiply << <blocks,threads>> > (a.GetDevPointer(),b.GetDevPointer(), result.GetDevPointer(), width);
	cudaError(cudaGetLastError());
}
__global__ void dGPUscalarmul(float* a, float scalar, float maxindex) {
	int indeks = blockIdx.x * blockDim.x + threadIdx.x;
	if (indeks < maxindex) {
		a[indeks] = scalar * a[indeks];
	}
}

void Math::GPUscalarmul(Matrix& a, float scalar) {

	int threads = 512;
	int blocks = (a.GetSize() * a.GetSize()) / threads + 1;
	int maxindex = a.GetSize() * a.GetSize();
	dGPUscalarmul<<<blocks, threads>>>(a.GetDevPointer(), scalar, maxindex);
}

__global__ void dGPUscalarmulstride(float* a, float scalar, float maxindex) {
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < maxindex; i += gridDim.x * blockDim.x) {
		a[i] = a[i] * scalar;
	}
}
void Math::GPUscalarmulstride(Matrix& a, float scalar) {
	int numSMs;
	cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
	int threads = 1024;
	int maxindex = a.GetSize() * a.GetSize();
	dGPUscalarmulstride << <8 * numSMs, threads >> > (a.GetDevPointer(), scalar, maxindex);

}
