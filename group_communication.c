#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define ROOT 0

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Bcast - distribute a message from root process to all other processes
    int num;
    if (rank == ROOT)
    {
        num = 100;
        printf("Process %d broadcasts %d\n", rank, num);
    }
    MPI_Bcast(&num, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (rank != ROOT)
    {
        printf("Process %d received %d from bcast\n", rank, num);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == ROOT) printf("\n");
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Scatter - distribute distinct messages from the root process to all other processes
    int nums[size];
    int i;
    if (rank == ROOT)
    {
        for (i = 0; i < size; i++)
        {
            nums[i] = 10 * (i + 1);
        }
    }
    MPI_Scatter(nums, 1, MPI_INT, &num, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    printf("Process %d received %d from scatter\n", rank, num);
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Gather - gather messages from all processes into the root process
    MPI_Gather(&num, 1, MPI_INT, nums, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT)
    {
        printf("Root process gathered data:");
        for (i = 0; i < size; i++)
        {
            printf(" %d", nums[i]);
        }
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Reduce - perform a reduction operation on values from all processes
    int sum = 0;
    MPI_Reduce(&num, &sum, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    if (rank == ROOT)
    {
        printf("Reduction result: %d\n", sum);
    }

    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 0 broadcasts 100
Process 1 received 100 from bcast
Process 2 received 100 from bcast
Process 3 received 100 from bcast

Process 0 received 10 from scatter
Process 1 received 20 from scatter
Process 2 received 30 from scatter
Process 3 received 40 from scatter
Root process gathered data: 10 20 30 40
Reduction result: 100
*/
