#include "GitrmSheath.hpp"
#include <netcdf.h>

// from https://docs.unidata.ucar.edu/netcdf-c/current/sfc__pres__temp__rd_8c_source.html
// Handle errors by printing an error message and exiting with a non-zero status.
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); return 2;}

int main( int argc, char* argv[] ) {

    Kokkos::initialize(argc,argv);{
        //for(int i = 100000; i < 2000000; i*=2)
        //{
        int factorOfMesh = atoi(argv[1]);

/*      //MPASMesh
        int retval;
        int ncid;
        if ((retval = nc_open(argv[1], NC_NOWRITE, &ncid)))
          ERR(retval);
        
*/
	//initialaize simple mesh 
        auto mesh = sheath::initializeTestMesh(factorOfMesh);//100000
        // inititalizew single particle
	// int rngSeed = 1010;
	// change seed (second argument) for different initial particle location
        auto part = sheath::initializeTestParticles(mesh);
        part.interpolateWachpress();	
        //part.interpolateWachpress();	
        //part.interpolateWachpress();	
        //part.interpolateWachpress();	
        //part.interpolateWachpress();
        assembly(mesh,part);
        assembly(mesh,part);
        assembly(mesh,part);
        assembly(mesh,part);
        assembly(mesh,part);
        //}
    }

    Kokkos::finalize();
    return 0;
}
