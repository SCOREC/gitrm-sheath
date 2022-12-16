#include "GitrmSheath.hpp"

int main( int argc, char* argv[] ) {

    Kokkos::initialize(argc,argv);{
	//initialaize simple mesh 
        auto mesh = sheath::initializeSimpleMesh();
        // inititalizew single particle
        auto part = sheath::initializeSingleParticle(mesh,1234);
	int npart = part.getTotalParticles();

        part.interpolateWachpress();	
    }

    Kokkos::finalize();
    return 0;
}
