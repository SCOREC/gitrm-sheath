#include "GitrmSheath.hpp"

void print_usage();

int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        if (argc != 4)
        {
            print_usage();
        }

        int Nel_x = atoi( argv[1] );
        int Nel_y = atoi( argv[2] );
        std::string nodeFile = argv[3];

        sheath::Mesh meshObj = sheath::initializeSheathMesh(Nel_x,Nel_y,nodeFile);
        sheath::Particles partObj = sheath::initializeParticles(1000000,meshObj);

        printf("Total nodes is %d\n",meshObj.getTotalNodes() );

    }
    Kokkos::finalize();

    return 0;
}

void print_usage()
{
    printf("Execute the code with the following command line arguments -- \n\n" );
    printf("\t ./install/bin/GitrmSheath_Demo Nel_x Nel_y  nodeCoordFile.dat \n\n\n");
    printf("\t Nel_x     \t\t Total Number of elements in hPIC mesh along the x1-direction \n");
    printf("\t Nel_y     \t\t Total Number of elements in hPIC mesh along the x2-direction \n");
    printf("\t \"nodeCoordFile.dat\" \t\t Location to file containing mapped node coordinates in the poloidal plane\n" );
    printf("  E.g.#1 \n\n");
    printf("    ./install/bin/GitrmSheath_Demo 32 56 node_coordinates.dat \n\n");
    printf("  E.g.#2 \n\n");
    printf("    ./install/bin/GitrmSheath_Demo 32 56 node_coordinates.dat \n\n");
    Kokkos::finalize();
    exit(0);
}
