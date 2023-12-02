#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "trank.h"
#include <time.h>

int main(int argc, char** argv) {
	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	// Seed the random number generator to get different results each time
	srand(time(NULL) * world_rank);

	float rand_num = rand() / (float)RAND_MAX;
	int rank;
	TMPI_Rank(&rand_num, &rank, MPI_FLOAT, MPI_COMM_WORLD);
	printf("Rank for %f on process %d - %d\n", rand_num, world_rank, rank);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
}