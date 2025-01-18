#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 256 // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1 // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0


__global__ void diff_eq(double *C, double *C_new, double *output, double *output_difmedio, int i){
		 	int x_position = threadIdx.x + blockIdx.x * blockDim.x;
			double up, down, left, right;
			int j;


			// calculo dos valores do halo do stencil
			up = (blockIdx.x > 0) ? C[threadIdx.x + (blockIdx.x - 1) * N] : 0.;
			down = (blockIdx.x < N-1) ? C[threadIdx.x + (blockIdx.x + 1) * N] : 0.;
			left = (threadIdx.x > 0) ? C[(threadIdx.x - 1) + blockIdx.x * N] : 0.;
			right = (threadIdx.x < N-1) ? C[(threadIdx.x + 1) + blockIdx.x * N] : 0.;

			//atualizacao da matriz com os novos valores
			C_new[x_position] = C[x_position] + D * DELTA_T * (
			(up + down + left + right - 4 * C[x_position]) / (DELTA_X * DELTA_X)
			);

			__syncthreads();

			//calculo do difmedio aqui

			if (threadIdx.x == 0) {
					 output[blockIdx.x] = 0;
					 for (j = 0; j < N; j ++){
								output[blockIdx.x] += fabs(C_new[x_position + j] - C[x_position + j]);
					 }
			}
			__syncthreads();
			if (x_position == 0) {
				output_difmedio[i] = 0;
				for (j = 0; j < N; j ++){
					output_difmedio[i] += output[j];
					if (i%100 == 0) printf("%g", output[j]);

				}
      	if (i%100 == 0) printf("%g", output_difmedio[i]);
			}

			C[x_position] = C_new[x_position];


}



int main(void)
{
	double *h_C = (double *)malloc(N*N*sizeof(double));
	double *h_C_new = (double *)malloc(N*N*sizeof(double));
	double *h_output = (double *)malloc(N*N*sizeof(double));
	double *h_output_difmedio = (double*)malloc(T*sizeof(double));

	double *d_C, *d_C_new, *d_output, *d_output_difmedio;

	// inicializacao da matriz
	for(int i=0; i< N*N; i++){
			 h_C[i] = 0.;
			 h_C_new[i] =0.;
	}

	int meio = ((N/2) * N) + N/2;

	h_C[meio] = 1.0;

	//alocacoes dos valores na GPU
	size_t size = N*N*sizeof(double);
	size_t prov = N *sizeof(double);

	cudaMalloc(&d_C, size);
	cudaMalloc(&d_C_new, size);
	cudaMalloc(&d_output, prov);
	cudaMalloc(&d_output_difmedio, T*sizeof(double));

	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_new, h_C_new, size, cudaMemcpyHostToDevice);



	// T iteracoes do stencil, sendo cada iteracao realizada por um kernel
	for(int k=0;k<T; k++) {
			 diff_eq<<<N,N>>>(d_C, d_C_new, d_output, d_output_difmedio, k);
	}
	cudaDeviceSynchronize();


	// Copia dos valores da GPU de volta para a CPU

	cudaMemcpy(h_output, d_C, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_output_difmedio, d_output_difmedio, T*sizeof(double), cudaMemcpyDeviceToHost);

	printf("\n\nConcentração do meio: %g\n", h_output[meio]);
  for(int i=0;i<T;i++){
				if (i%100 == 0) {
					printf("interacao %d - diferenca = %g\n", i, h_output_difmedio[i]/((N)*(N)));
				}
	}
	return 0;
}