#pragma once
#include "matrix.cuh"
#include "cudaerror.cuh"

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

class Math {
public:
	Math() {};
	~Math() {};
	void CPUadd(Matrix& a, Matrix& b, Matrix& result);
	void CPUsubtract(Matrix& a, Matrix& b, Matrix& result);
	void CPUtranspose(Matrix& a, Matrix &result);
	void CPUmultiply(Matrix& a, Matrix& b, Matrix& result);
	void CPUscalarmul(Matrix& a, float scalar);

	void GPUadd(Matrix& a, Matrix& b, Matrix& result);
	void GPUsubtract(Matrix& a, Matrix& b, Matrix& result);
	void GPUtranspose(Matrix& a, Matrix &result);
	void GPUmultiply(Matrix& a, Matrix& b, Matrix& result);
	void GPUscalarmul(Matrix& a, float scalar);
	void GPUscalarmulstride(Matrix& a, float scalar);


};