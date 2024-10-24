#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define ROOT 0
#define TAG 0
#define SENDER 1
#define RECEIVER 2
#define BUFFSIZE 50

int main(int argc, char** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char message[BUFFSIZE]; 

    // Broadcast message from root to all other processes
    if (rank == ROOT)
    {
        strcpy(message, "Hello from root!");
        printf("Process %d broadcasting message: %s\n", rank, message);
    }
    MPI_Bcast(&message, BUFFSIZE, MPI_CHAR, ROOT, MPI_COMM_WORLD);
    if (rank != ROOT)
    {
        printf("Process %d received broadcast message: %s\n", rank,  message);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Send messages to root from all other processes
    if (rank != ROOT)
    {
        snprintf(message, BUFFSIZE, "Hello from process %d", rank);
        MPI_Send(&message, BUFFSIZE, MPI_CHAR, ROOT, TAG, MPI_COMM_WORLD);
    }
    else
    {
        int i;
        for(i = 0; i < size; i++)
        {
            if (i == ROOT) continue;
            MPI_Recv(&message, BUFFSIZE, MPI_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Root process received message: %s\n", message);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Process 1 send message to process 2
    if (rank == SENDER)
    {
        snprintf(message, BUFFSIZE, "Message from process %d to %d", SENDER, RECEIVER);
        MPI_Send(message, BUFFSIZE, MPI_CHAR, RECEIVER, TAG, MPI_COMM_WORLD);
    }
    else if (rank == RECEIVER)
    {
        MPI_Recv(message, BUFFSIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received message: %s\n", RECEIVER, message);
    }
    

    MPI_Finalize();
    return 0;
}

/*
OUTPUT:
Process 0 broadcasting message: Hello from root!
Process 1 received broadcast message: Hello from root!
Process 2 received broadcast message: Hello from root!
Process 3 received broadcast message: Hello from root!
Root process received message: Hello from process 1
Root process received message: Hello from process 2
Root process received message: Hello from process 3
Process 2 received message: Message from process 1 to 2
*/
