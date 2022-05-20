#include "GitrmSheath.hpp"

int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        if (argc != 8)
        {
            print_usage();
        }

        int Nel_x = atoi( argv[1] );
        int Nel_y = atoi( argv[2] );
        std::string nodeFile = argv[3];

        sheath::initializeSheathMesh(Nel_x,Nel_y,nodeFile);

    }
    Kokkos::finalize();

    return 0;
}
