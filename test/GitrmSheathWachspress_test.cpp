#include "GitrmSheath.hpp"
int main( int argc, char* argv[] ) {

    Kokkos::initialize(argc,argv);{
	//initialaize simple mesh 
        auto mesh = sheath::initializeSimpleMesh();
        // inititalizew single particle
	int rngSeed = 1010;
	// change seed (second argument) for different initial particle location
        auto part = sheath::initializeSingleParticle(mesh,rngSeed);
	int npart = part.getTotalParticles();
        part.interpolateWachpress();	
    }

    Kokkos::finalize();
    sheath::initializeTestMesh();
    return 0;
}
