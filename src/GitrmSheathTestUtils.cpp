#include "GitrmSheathTestUtils.hpp"

namespace sheath{

Particles initializeParticles(int numParticles, Mesh meshObj){

    meshObj.computeFractionalElementArea();
    int Nel = meshObj.getTotalElements();
    auto fracArea = meshObj.getFractionalElementAreas();

    IntView initialParticlesPerElem("intial-particle-distribution",Nel);
    int adjustedTotParticles = 0;

    Kokkos::parallel_reduce("compute-part-per-elem", Nel, KOKKOS_LAMBDA(const int iel, int& update ){
        double partCount = fracArea(iel)*numParticles;
        initialParticlesPerElem(iel) = (int) partCount;
        update += initialParticlesPerElem(iel);
    },adjustedTotParticles);

    


    Particles p;
    return p;
}

}
