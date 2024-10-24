#include <mpi.h>
#include <stdio.h>

#define FAIL 5       
#define DETECT 1    
#define TAG 0        
#define NEXT_PROCESS ((rank+1)%size != FAIL) ? (rank+1)%size : (rank+2)%size

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Coordinator process fails
    if (rank == FAIL)
    {
        MPI_Finalize();
        return 0;
    }
    
    // Election initiation
    int max_rank, coordinator, next_process;
    if (rank == DETECT)
    {
        printf("Process %d detected failure of coordinator process %d and initiated an election.\n", rank, FAIL);
        MPI_Send(&rank, 1, MPI_INT, NEXT_PROCESS, TAG, MPI_COMM_WORLD);
    }

    // Receive election message and forward it around the ring
    MPI_Recv(&max_rank, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank != DETECT)
    {
        max_rank = (max_rank > rank) ? max_rank : rank;
        MPI_Send(&max_rank, 1, MPI_INT, NEXT_PROCESS, TAG, MPI_COMM_WORLD);
    }
    else
    {
        coordinator = max_rank;
    }

    // Broadcast the new coordinator to all processes
    MPI_Bcast(&coordinator, 1, MPI_INT, DETECT, MPI_COMM_WORLD);

    printf("Process %d: Coordinator is process %d\n", rank, coordinator);
    
    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 1 detected failure of coordinator process 5 and initiated an election.
Process 0: Coordinator is process 4
Process 1: Coordinator is process 4
Process 2: Coordinator is process 4
Process 4: Coordinator is process 4
Process 3: Coordinator is process 4
*/
