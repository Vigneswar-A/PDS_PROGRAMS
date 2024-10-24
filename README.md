# MPI Codes
| Experiment                                              | File Name                     |
|--------------------------------------------------------------|-------------------------------|
| Design chat server application with multiple clients         | [client_server.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/client_server.c)             |
| Program to implement mutual exclusion in distributed environment | [mutual_exclusion.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/mutual_exclusion.c)          |
| Program to implement group communication                     | [group_communication.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/group_communication.c)       |
| Program to implement clock synchronization                   | [clock_synchronization.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/clock_synchronization.c)     |
| Program to demonstrate leader election algorithm             | [leader_election.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/leader_election.c)           |
| Program to implement fault tolerance mechanism using Byzantine agreement | [byzantine_agreement.c](https://github.com/Vigneswar-A/PDS_PROGRAMS/blob/main/byzantine_agreement.c)       |

# Commands
| Command                                 | Description                                                       |
|-----------------------------------------|-------------------------------------------------------------------|
| `mpicc -o output_file source_file.c`   | Compiles a C source file using the MPI compiler wrapper.         |
| `mpirun -np <num_processes> ./output_file` | Executes the compiled MPI program with the specified number of processes. |
| `man <mpi_function_name>` | Shows documentation of the mpi function|

# Help
### MPI Functions

| Signature                         | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| `int MPI_Init(int *argc, char ***argv)` | Initializes the MPI environment.                                   |
| `int MPI_Comm_size(MPI_Comm comm, int *size)` | Determines the size of the group associated with a communicator.    |
| `int MPI_Comm_rank(MPI_Comm comm, int *rank)` | Determines the rank of the calling process in a communicator.       |
| `int MPI_Barrier(MPI_Comm comm)` | Synchronizes all processes in a communicator by blocking until all processes have called it. |
| `int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)` | Broadcasts a message from the root process to all other processes in the communicator. |
| `int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)` | Distributes distinct chunks of data from one process to all other processes. |
| `int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)` | Gathers data from all processes to a designated root process.        |
| `int MPI_Reduce(const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)` | Performs a reduction operation on data from all processes to a root process. |
| `int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)` | Sends a message to a specified destination process.                 |
| `int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)` | Receives a message from a specified source process.                |
| `int MPI_Finalize()`              | Cleans up the MPI environment and should be called before the program terminates. |

### MPI Constants and Types

| Constant/Type                     | Description                                                        |
|-----------------------------------|--------------------------------------------------------------------|
| `MPI_ANY_SOURCE`                  | A wildcard constant used in `MPI_Recv` to receive messages from any source. |
| `MPI_ANY_TAG`                     | A wildcard constant used in `MPI_Recv` to receive messages with any tag. |
| `MPI_CHAR`                        | MPI datatype representing a character.                             |
| `MPI_INT`                         | MPI datatype representing an integer.                              |
| `MPI_LONG`                        | MPI datatype representing a long integer.                          |
| `MPI_COMM_WORLD`                  | The communicator that includes all processes in the MPI program.   |
| `MPI_SOURCE`                      | An attribute in `MPI_Status` that contains the rank of the sending process. |
| `MPI_Status`                      | A structure that contains information about a received message.    |
| `MPI_STATUS_IGNORE`               | A constant indicating that the status of the message is not needed. |
| `MPI_SUM`                         | An operation for summing values in a reduction operation.          |
| `MPI_TAG`                         | An attribute in `MPI_Status` that indicates the message's tag.    |

###
