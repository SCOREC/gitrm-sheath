#include "GitrmSheath.hpp"

void print_usage();
void print_particle_state(sheath::Particles partObj, int iTime);


int main( int argc, char* argv[] )
{
    Kokkos::initialize( argc, argv );
    {
        if (argc != 7)
        {
            print_usage();
        }

        int Nel_x = atoi( argv[1] );
        int Nel_y = atoi( argv[2] );
        std::string nodeFile = argv[3];
        int numParticles = atoi(argv[4]);
        int rngSeed = atoi(argv[5]);
        double scale = atof(argv[6]);

        sheath::Mesh meshObj = sheath::initializeSheathMesh(Nel_x,Nel_y,nodeFile);
        sheath::Particles partObj = sheath::initializeParticles(numParticles,meshObj,rngSeed);
        numParticles = partObj.getTotalParticles();
        sheath::Vector2View disp = sheath::getRandDisplacements(numParticles,rngSeed,scale);

        // partObj.validateP2LAlgo();
        int numActiveParticles = partObj.computeTotalActiveParticles();
        int iTime = 0;
        printf("Total particles at T=%d is %d\n",iTime, numActiveParticles );
        print_particle_state(partObj,iTime);
        while(numActiveParticles > 0 && iTime<10){
            iTime++;
            partObj.T2LTracking(disp);
            numActiveParticles = partObj.computeTotalActiveParticles();
            print_particle_state(partObj,iTime);
            printf("Total particles at T=%d is %d\n",iTime, numActiveParticles);
        }
        // partObj.MacphersonTracking(disp);
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
    printf("\t Npart     \t\t Total particles initialized in domain\n" );
    printf("\t seed      \t\t Seed of random number generator\n" );
    printf("\t scale     \t\t Scaling factor for particle push\n" );
    printf("  E.g.#1 \n\n");
    printf("    ./install/bin/GitrmSheath_Demo 32 56 node_coordinates.dat 10000 1234 10.0\n\n");
    printf("  E.g.#2 \n\n");
    printf("    ./install/bin/GitrmSheath_Demo 32 56 node_coordinates.dat 100000 1234 0.5\n\n");
    Kokkos::finalize();
    exit(0);
}

void print_particle_state(sheath::Particles partObj, int iTime){
    int numParticles = partObj.getTotalParticles();
    auto xp = partObj.getParticlePostions();
    auto eID = partObj.getParticleElementIDs();
    auto status = partObj.getParticleStatus();
    printf("numParticles = %d\n",numParticles );

    auto h_xp = Kokkos::create_mirror_view(xp);
    Kokkos::deep_copy(h_xp,xp);
    auto h_eID = Kokkos::create_mirror_view(eID);
    Kokkos::deep_copy(h_eID,eID);
    auto h_status = Kokkos::create_mirror_view(status);
    Kokkos::deep_copy(h_status,status);

    FILE *part_file;
    char part_filename[30];
    sprintf(part_filename,"part_coords_t%d.dat",iTime);
    part_file = fopen(part_filename,"w");
    int skip = 1;
    for (int i=0; i<numParticles; i=i+skip){
        int part_active = h_status(i);
        int elemID = h_eID(i);
        fprintf(part_file, "%d %.5e %.5e %d\n", part_active, h_xp(i)[0], h_xp(i)[1], elemID);
    }

    fclose(part_file);
}
