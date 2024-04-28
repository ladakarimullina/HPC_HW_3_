#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>


typedef signed char cell_t;

//number of ghost cells on each side
const int ghost = 1;

void step( const cell_t *cur, cell_t *next, int ext_n )
{
    int i;
    const int LEFT = ghost;
    const int RIGHT = ext_n - ghost - 1;
    for (i = LEFT; i <= RIGHT; i++) {
        const cell_t east = cur[i-1];
        const cell_t center = cur[i];
        const cell_t west = cur[i+1];
        next[i] = ( (east && !center && !west) ||
                    (!east && !center && west) ||
                    (!east && center && !west) ||
                    (!east && center && west) );
    }
}
//Initial domain; all cells are 0, with the exception of a single cell in the middle of the domain
void init_domain( cell_t *cur, int ext_n )
{
    int i;
    for (i=0; i<ext_n; i++) {
        cur[i] = 0;
    }
    cur[ext_n/2] = 1;
}


//Dump the current state of the automaton to PBM file `out`

void dump_state( FILE *out, const cell_t *cur, int ext_n )
{
    int i;
    const int LEFT = ghost;
    const int RIGHT = ext_n - ghost - 1;

    for (i=LEFT; i<=RIGHT; i++) {
        fprintf(out, "%d ", cur[i]);
    }
    fprintf(out, "\n");
}

int main( int argc, char* argv[] )
{
    const char *outname = "rule30_parallel.pbm";
    FILE *out = NULL;
    int width = 120, nsteps = 120, s;
    cell_t *cur = NULL, *tmp;
    int my_rank, comm_sz;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    double t1, t2;


    if ( 0 == my_rank && argc > 3 ) {
        fprintf(stderr, "Usage: %s [width [nsteps]]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if ( argc > 1 ) {
        width = atoi(argv[1]);
    }

    if ( argc > 2 ) {
        nsteps = atoi(argv[2]);
    }

    if ( (0 == my_rank) && (width % comm_sz) ) {
        printf("The image width (%d) must be a multiple of comm_sz (%d)\n", width, comm_sz);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    const int ext_width = width + 2*ghost;

    if ( 0 == my_rank ) {
        out = fopen(outname, "w");
        if ( !out ) {
            fprintf(stderr, "FATAL: Cannot create %s\n", outname);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(out, "P1\n");
        fprintf(out, "# Produced by mpi-rule30\n");
        fprintf(out, "%d %d\n", width, nsteps);

        cur = (cell_t*)malloc( ext_width * sizeof(*cur) ); assert(cur != NULL);
        init_domain(cur, ext_width);
    }

    const int rank_next = (my_rank + 1) % comm_sz;
    const int rank_prev = (my_rank - 1 + comm_sz) % comm_sz;

    const int local_width = width / comm_sz;
    const int local_ext_width = local_width + 2*ghost;

    cell_t *local_cur = (cell_t*)malloc(local_ext_width * sizeof(*local_cur)); assert(local_cur != NULL);
    cell_t *local_next = (cell_t*)malloc(local_ext_width * sizeof(*local_next)); assert(local_next != NULL);

    const int LEFT_GHOST = 0;
    const int LEFT = LEFT_GHOST + ghost;

    const int LOCAL_LEFT_GHOST = 0;
    const int LOCAL_LEFT = LOCAL_LEFT_GHOST + ghost;
    const int LOCAL_RIGHT = local_ext_width - 1 - ghost;
    const int LOCAL_RIGHT_GHOST = LOCAL_RIGHT + ghost;

    MPI_Scatter( &cur[LEFT],            /* sendbuf      */
                 local_width,           /* sendcount    */
                 MPI_CHAR,              /* datatype     */
                 &local_cur[LOCAL_LEFT],/* recvbuf      */
                 local_width,           /* recvcount    */
                 MPI_CHAR,              /* datatype     */
                 0,                     /* root         */
                 MPI_COMM_WORLD
                 );

    for (s=0; s<nsteps; s++) {

        if ( 0 == my_rank ) {
            t1 = MPI_Wtime();
            dump_state(out, cur, ext_width);
        }

        MPI_Sendrecv( &local_cur[LOCAL_RIGHT], 
                      ghost,             
                      MPI_CHAR,         
                      rank_next,        
                      0,                
                      &local_cur[LOCAL_LEFT_GHOST],
                      ghost,             
                      MPI_CHAR,         
                      rank_prev,        
                      0,               
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );


        MPI_Sendrecv( &local_cur[LOCAL_LEFT], /* sendbuf      */
                      ghost,             /* sendcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_prev,        /* dest         */
                      0,                /* sendtag      */
                      &local_cur[LOCAL_RIGHT_GHOST], /* recvbuf */
                      ghost,             /* recvcount    */
                      MPI_CHAR,         /* datatype     */
                      rank_next,        /* source       */
                      0,                /* recvtag      */
                      MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE
                      );

        step( local_cur, local_next, local_ext_width );


        MPI_Gather( &local_next[LOCAL_LEFT],
                    local_width,        
                    MPI_CHAR,          
                    &cur[LEFT],         
                    local_width,        
                    MPI_CHAR,           
                    0,                  
                    MPI_COMM_WORLD
                    );

        //swap current and next domain 
        tmp = local_cur;
        local_cur = local_next;
        local_next = tmp;
    }

    t2 = MPI_Wtime();

    //free memory
    free(local_cur);
    free(local_next);
    free(cur);

    if ( 0 == my_rank ) {
        fclose(out);
        printf("elapsed time = %f\n", t2 - t1);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
