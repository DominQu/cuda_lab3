#include <iostream>
#include "matrix.cuh"
#include "math.cuh"


const int MAX_SIZE = 3072;
const int OMEGA = 5;
const int MIKRO = 3;

void mathtest(Math& math, Matrix& a, Matrix& b, Matrix& x, int size) {
	// A*B + u*A^T + A - w*B
	// A*B
	Matrix ab(size, true, "AB");
	math.GPUmultiply(a, b, ab);
	// u*A^T
	Matrix at(size, true, "AT");
	math.GPUtranspose(a, at);
	math.GPUscalarmul(at, MIKRO);
	//w*B
	math.GPUscalarmul(b, OMEGA);
	//cudaDeviceSynchronize();
	// A*B + u*A^T
	math.GPUadd(ab, at, ab);
	// A - w*B
	math.GPUsubtract(a, b, x);
	//cudaDeviceSynchronize();
	// last addition
	math.GPUadd(ab, x, x);
}

void test(int reps, Math &math) {
	std::fstream outFile;
	outFile.open("GPUscalarmulstride.txt", std::ios::out);

	Matrix a(MAX_SIZE, false, "A");
	Matrix b(MAX_SIZE, false, "B");
	Matrix x(MAX_SIZE, true, "X");
	if (!outFile) {
		std::cout << "Couldn't open file!" << std::endl;
	}
	else {
		for (int j = 32; j < MAX_SIZE; j+=32) {
			a.SetSize(j);
			b.SetSize(j);
			x.SetSize(j);

			cudaEvent_t start, stop;
			float time;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);
			for (int i = 0; i < reps; i++) {
				//mathtest(math, a, b, x, j);
				math.GPUscalarmul(a, 1.05);
				cudaDeviceSynchronize();
			}

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			outFile << std::to_string(j);
			outFile << ",";
			outFile << std::to_string(time / reps);
			outFile << "\n";
			if (!(j % 128)) {
				std::cout << "Current matrix size: " << j << " current time: " << time / reps << std::endl;
			}
		}
		std::cout << "File created" << std::endl;
		outFile.close();
	}
}

int main() {
	
	Math math;
	//test(5, math);
	int size = 512;
	Matrix a(size, false, "A");
	Matrix b(size, false, "B");
	Matrix x(size, true, "X");
	mathtest(math, a, b, x, size);

	//math.GPUmultiply(a, b, x);
	//math.GPUtranspose(a, x);
	//math.GPUscalarmul(a,3.5);
	a.SaveToFile();
	b.SaveToFile();
	x.SaveToFile();
	return 0;
}
