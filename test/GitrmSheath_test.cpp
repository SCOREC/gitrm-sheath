#include "GitrmSheath.hpp"

int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        sheath::Mesh m = sheath::dummy();
    }
    Kokkos::finalize();

    return 0;
}
