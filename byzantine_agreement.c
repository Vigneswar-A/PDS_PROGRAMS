#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define FAULTY 3
#define CORRECT 3

int main(int argc, char** argv) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Generate random value for faulty process and correct value for other processes
    int received_value[size];
    srand(time(NULL) + rank);
    int value = (rank == FAULTY) ? rand() % 10 : CORRECT;

    // Broadcast all value to all processes
    MPI_Allgather(&value, 1, MPI_INT, received_value, 1, MPI_INT, MPI_COMM_WORLD);

    // Compute majority value - Boyer-Moore Majority Voting Algorithm
    int majority_element = 0;
    int majority_count = 0;
    int i;
    for (i = 0; i < size; i++) 
    {
        if (majority_count == 0) 
        {
            majority_count += 1;
            majority_element = received_value[i];
        } 
        else if (received_value[i] == majority_element) 
            majority_count += 1;
        else
            majority_count -= 1;   
    }

    // If value differs from majority, the process is faulty
    if (rank != FAULTY) 
    {
        for (i = 0; i < size; i++) 
            if (received_value[i] != majority_element) 
                printf("Process %d: %d is the faulty process\n", rank, i);
    }

    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 0: 3 is the faulty process
Process 2: 3 is the faulty process
Process 4: 3 is the faulty process
Process 1: 3 is the faulty process
*/
