
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include "device_functions.h"
#include "cuda_runtime_api.h"
#include <iostream>
#include "helper_cuda.h"
#include <iostream>
#include <fstream>
#include <ctime> 


long readList(long**);
void mergesort(long*, long, dim3, dim3);
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpuAbajoArriba(long*, long*, long, long, long);




int main(int argc, char** argv) {

	//
	//Inicializamos los hilos y bloques que deseamos usar
	//En nuestro caso se inicializan una cantidad minima
	//Ya que la memoria es suficiente para una ejecucion rapida del algoritmo
	//
	dim3 threadsPerBlock;
	dim3 blocksPerGrid;

	threadsPerBlock.x = 32;
	threadsPerBlock.y = 1;

	blocksPerGrid.x = 8;
	blocksPerGrid.y = 1;

	unsigned t0, t1;
	//
	// Cualquier tecla (RUN) que ejecuta el comando.
	//
	// Lee los numeros introducidos por consola
	//
	
	
	long* data;
	long size = readList(&data);
	if (!size) return -1;


	//Se inicializa la variable con la hora antes de la ejecucion
	t0 = clock();
	//Metodo merge-sort, le pasamos los datos, su tamaño y los hilos y bloques que deseamos usar
	mergesort(data, size, threadsPerBlock, blocksPerGrid);
	//Se inicializa otra variable con la hora despues de que finalize merge sort
	t1 = clock();

	//
	// Imprimimos la lista Ordenada
	//

	
	//
	//Dado que lo que nos interesa es conocer el tiempo del metodo mergesort
	//no se tiene en cuenta el tiempo de impresion.
	//
	std::cout << "\n";
	double time = (double(t1 - t0) / CLOCKS_PER_SEC);
	std::cout << size;
	std::cout << "Execution Time: " << time << std::endl;
	system("Pause");

}

//
//Metodo Mergesort
//
void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

	//
	// Creamos dos Arrays en la GPU
	// En los cuales iremos accediendo durante la ejecucion
	//

	long* A_Datos;
	long* A_Ordenacion;
	dim3* D_threads;
	dim3* D_blocks;


	checkCudaErrors(cudaMalloc((void**)&A_Datos, size * sizeof(long)));
	checkCudaErrors(cudaMalloc((void**)&A_Ordenacion, size * sizeof(long)));
		std::cout << "Creando Arrays en GPU....\n";

	// Copiamos los datos introducidos en el primer Array
	checkCudaErrors(cudaMemcpy(A_Datos, data, size * sizeof(long), cudaMemcpyHostToDevice));

	//
	// Reservamos la memoria en la GPU
	//
	checkCudaErrors(cudaMalloc((void**)&D_threads, sizeof(dim3)));
	checkCudaErrors(cudaMalloc((void**)&D_blocks, sizeof(dim3)));

	std::cout << "Reservando memoria en la GPU....\n";

	checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));


	long* A = A_Datos;
	long* B = A_Ordenacion;

	long nThreads = threadsPerBlock.x * threadsPerBlock.y *	blocksPerGrid.x * blocksPerGrid.y;

	//
	// Cortamos la lista en Slices
	//

	for (int width = 2; width < (size << 1); width <<= 1) {
		long slices = size / ((nThreads)* width) + 1;

		// Con esta funcion realizamos la llamada a la GPU

		gpu_mergesort << <blocksPerGrid, threadsPerBlock >> >(A, B, size, width, slices, D_threads, D_blocks);

	//Para ahorrar tiempo, cambiamos los vectores, en vez de copiarlos otra vez
	//Dado que segun Nvidia, en vectores de gran tamaño esto supone un ahorro de tiempo.
		A = A == A_Datos ? A_Ordenacion : A_Datos;
		B = B == A_Datos ? A_Ordenacion : A_Datos;
	}

	//
	// Devolvemos los valores
	//

	checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));


	// Liberamos memoria
	checkCudaErrors(cudaFree(A));
	checkCudaErrors(cudaFree(B));
	std::cout << "Liberando memoria en la GPU por si los Arrays A y B tienen valores....\n";
}

//
// Debido a que la funcion __syncronize no funciona 
// Usamos este metodo para localizar la id de cada hilo que ejecutamos
//
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
	int x;
	return threadIdx.x +
		threadIdx.y * (x = threads->x) +
		threadIdx.z * (x *= threads->y) +
		blockIdx.x  * (x *= threads->z) +
		blockIdx.y  * (x *= blocks->z) +
		blockIdx.z  * (x *= blocks->y);
}

//
// Aplicamos mergsort a nuestros datos
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
	unsigned int idx = getIdx(threads, blocks);
	long emp = width*idx*slices,
		mitad_flag,
		final;

	for (long slice = 0; slice < slices; slice++) {
		if (emp >= size)
			break;

		mitad_flag = min(emp + (width >> 1), size);
		final = min(emp + width, size);
		gpuAbajoArriba(source, dest, emp, mitad_flag, final);
		emp += width;
	}
}

//
//Metodo mergsort de ordenacion, a cada hilo se le pasa, sus datos, donsde se debe guardar
// donde empieza, la flag de mitad, y el final.
//
__device__ void gpuAbajoArriba(long* source, long* dest, long emp, long mitad_flag, long final) {
	long i = emp;
	long j = mitad_flag;
	for (long k = emp; k < final; k++) {
		if (i < mitad_flag && (j >= final || source[i] < source[j])) {
			dest[k] = source[i];
			i++;
		}
		else {
			dest[k] = source[j];
			j++;
		}
	}
}



// Funcion para leer los datos por consola

typedef struct {
	int v;
	void* sig_valor;
} Linknodo;



long readList(long** list) {

	long v, size = 0;
	Linknodo* nodo = 0;
	Linknodo* prim = 0;
	while (std::cin >> v) {
		Linknodo* sig_valor = new Linknodo();
		sig_valor->v = v;
		if (nodo)
			nodo->sig_valor = sig_valor;
		else
			prim = sig_valor;
		nodo = sig_valor;
		size++;
	}


	if (size) {
		*list = new long[size];
		Linknodo* nodo = prim;
		long i = 0;
		while (nodo) {
			(*list)[i++] = nodo->v;
			nodo = (Linknodo*)nodo->sig_valor;
		}

	}

	return size;
}
