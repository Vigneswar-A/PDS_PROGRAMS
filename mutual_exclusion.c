#include <mpi.h>
#include <stdio.h>

#define REQUEST 1
#define OK 2
#define RELEASE 3
#define ROOT 0

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // All processes request every other process to enter critical section
    if (rank == ROOT)
        printf("Process %d requesting critical section\n", rank);

    int i;
    for (i = 0; i < size; i++)
    {
        if (i != rank)
            MPI_Send(NULL, 0, MPI_CHAR, i, REQUEST, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // Each process replies to requests if it is not in critical section or has lower priority
    int replies = 0;
    MPI_Status status;
    
    while (replies < size - 1)
    {
        MPI_Recv(NULL, 0, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == REQUEST)
        {
            if (rank == ROOT)
                printf("Process %d received REQUEST from process %d\n", rank, status.MPI_SOURCE);
            
            if (status.MPI_SOURCE < rank)
                MPI_Send(NULL, 0, MPI_CHAR, status.MPI_SOURCE, OK, MPI_COMM_WORLD);
        }
        else if (status.MPI_TAG == OK)
        {
            if (rank == ROOT)
                printf("Process %d received OK from process %d\n", rank, status.MPI_SOURCE);
            
            replies++;
        }
        else if (status.MPI_TAG == RELEASE)
        {
            if (status.MPI_SOURCE == ROOT)
                printf("Process %d received RELEASE from process %d\n", rank, status.MPI_SOURCE);
            
            replies++;
        }
    }

    // Enter the critical section
    if (rank == ROOT)
        printf("Process %d in critical section\n", rank);
    
    // Release the critical section
    if (rank == ROOT)
        printf("Process %d releasing critical section\n", rank);

    for (i = 0; i < size; i++)
    {
        if (i != rank)
            MPI_Send(NULL, 0, MPI_CHAR, i, RELEASE, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 0 requesting critical section
Process 0 received REQUEST from process 1
Process 0 received REQUEST from process 2
Process 0 received REQUEST from process 3
Process 0 received OK from process 1
Process 0 received OK from process 2
Process 0 received OK from process 3
Process 0 in critical section
Process 0 releasing critical section
Process 1 received RELEASE from process 0
Process 2 received RELEASE from process 0
Process 3 received RELEASE from process 0
*/
