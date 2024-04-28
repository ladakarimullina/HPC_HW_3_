
#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>


template <typename T>
void print_players(const std::vector<T> &players) {

    for(int i = 0; i < players.size(); ++i) {

        printf("Player %d ", players[i]);
    }
    printf(": %ld passes", players.size() - 1);
    printf("\n");
}

int main(int argc, char **argv) {
    const int passes = 10;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    srand(rank + 43); 

    std::vector<int> players;
    players.reserve(passes);

    int current = 0; // since that Rank 0 starts the game
    int next = 0;

    int number_of_balls_passed = 0;

    bool play = true;

    while (play) {

        if (rank == current) {
            players.push_back(rank);

            if (players.size() == passes + 1) {
                int stop = 1;
                for(int index = 0; index < size; ++index) {
                    if (index != rank) {
                        MPI_Send(&stop, 1, MPI_INT, index, 1, MPI_COMM_WORLD);
                    }
                }

                print_players(players);

                break;
            }


            next = current;
            while (next == current) next = rand() % size;
            current = next;
            print_players(players);

            int win = MPI_Ssend(players.data(), players.size(), MPI_INT, next, 0, MPI_COMM_WORLD);

            if (win != MPI_SUCCESS) {
                printf("Error while sending from process %d to process %d: code %d\n", rank, next, win);
            }

        } 
        else {
            int data[passes];
            int swin = MPI_Recv(&data, passes, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            current = rank;
            int count;
            MPI_Get_count(&status, MPI_INT, &count);

            players.clear();
            for(int index = 0; index < count; ++index) {
                players.push_back(data[index]);
            }

            if (status.MPI_TAG == 1) {
                play = false;
            }
        }
    }
    MPI_Finalize();
}
