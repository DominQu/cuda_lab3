#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <random>

const int POINT_RANGE = 10;

class Matrix {
private:
	thrust::host_vector<float> dataCPU;
	thrust::device_vector<float> dataGPU;
	int size;
	int externsize;
	const char* name;
	void GenerateData(thrust::host_vector<float>& CPUmatrix,
		                                    thrust::device_vector<float>& GPUmatrix, 
		                                    int num_points);
public:
	Matrix(int size, bool empty, const char *name);
	~Matrix() {};
	
	thrust::host_vector<float> getCPUdata();
	thrust::device_vector<float> getGPUdata();

	const char* GetName();
	int GetSize();
	float* GetDevPointer();

	void SetSize(int size);

	void SaveToFile();

	float& at(int indeks);

};