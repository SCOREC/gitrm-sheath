#include "GitrmSheath.hpp"
int main( int argc, char* argv[] ) {

    Kokkos::initialize(argc,argv);{
        //for(int i = 100000; i < 2000000; i*=2)
        //{
        int factorOfMesh = atoi(argv[1]);
	//initialaize simple mesh 
        auto mesh = sheath::initializeTestMesh(factorOfMesh);//100000
        // inititalizew single particle
	// int rngSeed = 1010;
	// change seed (second argument) for different initial particle location
        auto part = sheath::initializeTestParticles(mesh);
        part.interpolateWachpress();	
        part.interpolateWachpress();	
        part.interpolateWachpress();	
        part.interpolateWachpress();	
        part.interpolateWachpress();	
        sheath::assembly(mesh);
        //}
    }

    Kokkos::finalize();
    return 0;
}
