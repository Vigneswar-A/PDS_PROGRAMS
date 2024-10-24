#include <mpi.h>
#include <stdio.h>
#include <time.h>

#define ROOT 0

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get local time of each process
    time_t local_time = time(NULL);
    printf("Process %d local time: %ld\n", rank, local_time);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Compute average time of all the processes
    long avg_time;
    long time_sum;
    int i;
    MPI_Reduce(&local_time, &time_sum, 1, MPI_LONG, MPI_SUM, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT)
    {
        avg_time = (time_sum / size);
        printf("Average time: %ld\n", avg_time);
    }
    MPI_Bcast(&avg_time, 1, MPI_LONG, ROOT, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // Adjust to the average time
    printf("Process %d adjusted time: %ld\n", rank, avg_time);

    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 0 local time: 1729787894
Process 1 local time: 1729787894
Process 2 local time: 1729787894
Process 3 local time: 1729787894
Average time: 1729787894
Process 0 adjusted time: 1729787894
Process 1 adjusted time: 1729787894
Process 2 adjusted time: 1729787894
Process 3 adjusted time: 1729787894
*/
