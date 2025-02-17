#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#define SIZE 4000
#define T 500
#define D 0.1 // Coeficiente de difusão
#define DELTA_T 0.01
#define DELTA_X 1.0

double ** create_matrixes(double middle_value){
    double ** C = malloc(SIZE*sizeof(double));


    for(int k=0; k<SIZE; k++)
        C[k] = malloc(SIZE*sizeof(double));



    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            C[i][j] = 0.;
            
    C[SIZE/2][SIZE/2] = middle_value;

    return C;
}

double * create_linear_matrixes(double middle_value){
    double * C = malloc(SIZE*SIZE*sizeof(double));

    for(int i=0; i<SIZE*SIZE; i++){
        C[i] = 0.;
    }
         
    C[((SIZE/2) * SIZE) + SIZE/2] = middle_value;

    return C;
}

void free_matrix(double ** c){

    for(int k=0; k<SIZE; k++){
        free(c[k]);
    }

    free(c);
}

void print_matrix(double ** c, int lines){
    for(int i=0; i<SIZE; i++){
        printf("\n");
        for(int j=0; j<SIZE; j++)
            printf(" %lf ", c[i][j]);
     }       
}

void print_linear_matrix(double *c, int lines){
    for(int i=0; i<lines*SIZE; i++){
        if(i%SIZE == 0) printf("\n");
        printf(" %lf ", c[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[]){


    int num_procs, my_id;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    
    int chunk_size = (SIZE/(num_procs));
    double *C = NULL;
    double *MATRIZ = malloc(chunk_size * SIZE * sizeof(double));
    double *temp = malloc(chunk_size * SIZE * sizeof(double));
    double difmedio_global = 0.;
    
    memset(temp, 0, chunk_size * SIZE * sizeof(double));
    if(my_id == 0){
        
        C = create_linear_matrixes(1.0);
        
    }
    
    
    for (int t = 0; t < T; t++) {
        MPI_Scatter(C, chunk_size * SIZE, MPI_DOUBLE, MATRIZ, chunk_size * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        double up_border[SIZE] = {0.}, down_border[SIZE] = {0.};
        MPI_Status status;
        
        if (my_id > 0) {
            MPI_Recv(up_border, SIZE, MPI_DOUBLE, my_id - 1, 0, MPI_COMM_WORLD, &status);
            MPI_Send(MATRIZ, SIZE, MPI_DOUBLE, my_id - 1, 0, MPI_COMM_WORLD);
        }
        if (my_id < num_procs - 1) {
            MPI_Send(&MATRIZ[(chunk_size - 1) * SIZE], SIZE, MPI_DOUBLE, my_id + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(down_border, SIZE, MPI_DOUBLE, my_id + 1, 0, MPI_COMM_WORLD, &status);
        }
        
        for (int i = 0; i < chunk_size * SIZE; i++) {
            double up = (i >= SIZE) ? MATRIZ[i - SIZE] : up_border[i % SIZE];
            double down = (i < SIZE * (chunk_size - 1)) ? MATRIZ[i + SIZE] : down_border[i % SIZE];
            double left = (i % SIZE != 0) ? MATRIZ[i - 1] : 0.;
            double right = ((i + 1) % SIZE != 0) ? MATRIZ[i + 1] : 0.;
            
            temp[i] = MATRIZ[i] + D * DELTA_T * (
                (up + left + right + down - 4*MATRIZ[i]) / (DELTA_X*DELTA_X)
            );
            
        }
        
        double difmedio_local = 0.;
        for (int i = 0; i < chunk_size*SIZE; i++)
        {
            difmedio_local += fabs(temp[i] - MATRIZ[i]);
        }
        
        MPI_Reduce(&difmedio_local, &difmedio_global,  1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        MPI_Gather(temp, chunk_size * SIZE, MPI_DOUBLE, C, chunk_size * SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        
        // MPI_Barrier(MPI_COMM_WORLD);
        
        if(my_id == 0 && t%100 == 0) printf("\n\nITERACAO %d DIFMEDIO: %g\n\n", t, difmedio_global/(SIZE*SIZE));
    }
    
        
        if(my_id == 0){
            
            printf("CONCENTRACAO FINAL NO CENTRO: %f", C[((SIZE/2) * SIZE) + SIZE/2]);
        
        }

    free(MATRIZ);
    free(temp);
    free(C);
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();


    return 0;

}


