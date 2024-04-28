#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

int round_double_to_int(double num) {
    int a = num;
    if (num - a >= 0.5) return a + 1;

    return a;
}

int* create_byte_sized_message(int bytes) {
    int size = round_double_to_int(bytes/sizeof(int));

    int* message = new int[size];
    message[0] = 0;

    return message;
}

template <typename T>
void free_allocated_memory(T* memo) {
    delete[] memo;
}

void play_ping_pong(int message_size_bytes, int passes, MPI_Status* status, int rank, int size) {
    int message_length = round_double_to_int(message_size_bytes/sizeof(int));
    int* message = create_byte_sized_message(message_size_bytes);

    int current_rank = 0; 
    int next_rank = 0;

    bool play = true;

    while (play) {

        if (rank == current_rank) {
            if (message[0] == passes) {
                int stop = 1;
                for(int index = 0; index < size; ++index) {
                    if (index != rank) {
                        MPI_Send(&stop, 4, MPI_BYTE, index, 1, MPI_COMM_WORLD);
                    }
                }
                break;
            }

            next_rank = current_rank;
            while (next_rank == current_rank) next_rank = rand() % size;

            current_rank = next_rank;
            MPI_Send(message, message_size_bytes, MPI_BYTE, next_rank, 0, MPI_COMM_WORLD);
        } 
        else {

            MPI_Recv(message, message_size_bytes, MPI_BYTE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, status);
            current_rank = rank;

            message[0] += 1;

            if (status->MPI_TAG == 1) {
                play = false;
            }
        }
    }

    free_allocated_memory(message);
}

int main(int argc, char **argv) {
    const int passes = 10000; 
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    srand(rank + 43);

    double startTime, endTime, elapsedTime;

    if (rank == 0) {
        printf("Number of passes\tSize\tIterations\tTotal time\tTime per message\tBandwidth\n");
    }

    for (int messageSize = 1024; messageSize <= 1000000; messageSize *= 2) {

        int iterations = 10000000 / messageSize; 
        
        if (rank == 0) startTime = MPI_Wtime();

        for (int i = 0; i < iterations; ++i) {
            play_ping_pong(messageSize, passes, &status, rank, size);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            endTime = MPI_Wtime();
            elapsedTime = endTime - startTime;
            double timePerMessage = elapsedTime / iterations;
            double bandwidth = (messageSize * iterations) / (elapsedTime * 1024 * 1024); // Bandwidth in MB/s

            printf("%d\t%d\t%d\t%.6f\t%.6f\t%.2f\n", passes, messageSize, iterations, elapsedTime, timePerMessage, bandwidth);
        }
    }

    MPI_Finalize();
    return 0;
}
