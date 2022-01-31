#include "matrix.cuh"


void Matrix::GenerateData(thrust::host_vector<float> &CPUmatrix,
                               thrust::device_vector<float> &GPUmatrix, 
                               int num_points) {

    // setup random generator
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // pushback random points to vector
    for (int i = 0; i < num_points; i++) {
        float x = dist(eng) * POINT_RANGE ;
        CPUmatrix[i] = x;
        GPUmatrix[i] = x;
    }
}

Matrix::Matrix(int size, bool empty, const char *name) : dataCPU(thrust::host_vector<float>(size* size)),
                            dataGPU(thrust::device_vector<float>(size*size)),
                            size(size),
                            externsize(size),
                            name(name)
{
    if (empty != true) {
        GenerateData(this->dataCPU, this->dataGPU, size*size);
    }
}

thrust::host_vector<float> Matrix::getCPUdata() {
    return this->dataCPU;
}

thrust::device_vector<float> Matrix::getGPUdata() {
    return this->dataGPU;
}

const char* Matrix::GetName() {
    return this->name;
}

int Matrix::GetSize() {
    return this->externsize;
}

float* Matrix::GetDevPointer() {
    return this->dataGPU.data().get();
}

void Matrix::SetSize(int size) {
    this->externsize = size;
}
void Matrix::SaveToFile() {
    std::fstream outFile;
    outFile.open((std::string)this->GetName() + ".txt", std::ios::out);
    if (!outFile) {
        std::cout << "Error" << std::endl;
    }
    else {
        for (int i = 0; i < this->size; i++) {
            for (int j = 0; j < this->size; j++) {
                int indeks = i * this->size + j;
                outFile << std::to_string(dataGPU[indeks]);
                outFile << ";";
            }
            outFile << "\n";
        }
        std::cout << "File created" << std::endl;
        outFile.close();
    }
}

float& Matrix::at(int indeks) {
    return this->dataCPU[indeks];
}