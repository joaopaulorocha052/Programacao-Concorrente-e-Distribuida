#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#define N 5000  // Tamanho da grade
#define T 500 // Número de iterações no tempo
#define D 0.1  // Coeficiente de difusão
#define NUM_TESTS 3
#define DELTA_T 0.01
#define DELTA_X 1.0

void diff_eq(double **C, double **C_new, int num_threads) { //diff_eq(double C[N][N], double C_new[N][N]) {
    omp_set_num_threads(num_threads);
    for (int t = 0; t < T; t++) {
      #pragma omp parallel for
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                C_new[i][j] = C[i][j] + D * DELTA_T * (
                    (C[i+1][j] + C[i-1][j] + C[i][j+1] + C[i][j-1] - 4 * C[i][j]) / (DELTA_X * DELTA_X)
                );
            }
        }
        // Atualizar matriz para a próxima iteração
        double difmedio = 0.;
        #pragma omp parallel for reduction(+ : difmedio)
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                difmedio += fabs(C_new[i][j] - C[i][j]);
                C[i][j] = C_new[i][j];
            }
        }
        if ((t%100) == 0)
          printf("interacao %d - diferenca=%g\n", t, difmedio/((N-2)*(N-2)));
    }
}

int main() {

    // variaveis de tempo
    double start_time, end_time, execution_time;
    // Arquivo de resultados

    FILE * file;
    // printf("%d", omp_get_max_threads());
    // Concentração inicial
    double **C = (double **)malloc(N * sizeof(double *));
    if (C == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
    
    for (int i = 0; i < N; i++) {
      C[i] = (double *)malloc(N * sizeof(double));
      if (C[i] == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
      }
    }
    double **C_new = (double **)malloc(N * sizeof(double *));

    // Concentração para a próxima iteração
    if (C_new == NULL) {
      fprintf(stderr, "Memory allocation failed\n");
      return 1;
    }
    for (int i = 0; i < N; i++){
    
        C_new[i] = (double *)malloc(N * sizeof(double));
        if (C_new[i] == NULL){
        
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
    }


    
    file = fopen("results_new.txt", "w");

    // Executar as iterações no tempo para a equação de difusão
    fprintf(file, "NumThreads,Time,difmedio\n");
    for (int num_test = 0; num_test < NUM_TESTS; num_test++){
        printf("iniciando teste: %d\n", num_test+1);
    
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < N; j++) {
            C[i][j] = 0.;
          }
        }
        
        for (int i = 0; i < N; i++){
        
            for (int j = 0; j < N; j++){
            
                C_new[i][j] = 0.;
            }
        }

        // Inicializar uma concentração alta no centro
        C[N / 2][N / 2] = 2.5;
        printf("Thread 1:\n");
        start_time = omp_get_wtime();
        diff_eq(C, C_new, 1);
        end_time = omp_get_wtime();

        execution_time = (end_time-start_time);
        printf("%lf\n", execution_time);

        fprintf(file, "%d,%lf,%lf\n",1, execution_time, C[N/2][N/2]);
        for (int num_thread = 2; num_thread <= 12; num_thread+=2){
        
        	for (int i = 0; i < N; i++) {
		      for (int j = 0; j < N; j++) {
		        C[i][j] = 0.;
		      }
		    }
        
		    for (int i = 0; i < N; i++){
		    
		        for (int j = 0; j < N; j++){
		        
		            C_new[i][j] = 0.;
		        }
		    }
		    C[N / 2][N / 2] = 2.5;
        	
        	printf("Thread %d:\n", num_thread);
            start_time = omp_get_wtime();
            diff_eq(C, C_new, num_thread);
            end_time = omp_get_wtime();

            execution_time = (end_time-start_time);
            printf("%lf\n", execution_time);

            fprintf(file, "%d,%lf,%lf\n",num_thread, execution_time, C[N/2][N/2]);
        }
    }
    
    fclose(file);
    // Exibir resultado para verificação
    //printf("%lf\n", C[(N/2)-1][(N/2)+1]);
    //printf("Concentração final no centro: %f\n", C[N/2][N/2]);
    return 0;
}