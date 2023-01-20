#include "GitrmSheath.hpp"
int main( int argc, char* argv[] ) {

    Kokkos::initialize(argc,argv);{
	//initialaize simple mesh 
        auto mesh = sheath::initializeTestMesh();
        // inititalizew single particle
	// int rngSeed = 1010;
	// change seed (second argument) for different initial particle location
        auto part = sheath::initializeTestParticles(mesh);
        part.interpolateWachpress();	
    }

    Kokkos::finalize();
    return 0;
}
