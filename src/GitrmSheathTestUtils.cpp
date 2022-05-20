#include "GitrmSheathTestUtils.hpp"

namespace sheath{

Particles initializeParticles(int numParticles, Mesh meshObj, unsigned int rngSeed){

    meshObj.computeFractionalElementArea();
    int Nel = meshObj.getTotalElements();
    auto fracArea = meshObj.getFractionalElementAreas();
    printf("Total mesh volume is %2.5e\n",meshObj.getTotalArea() );
    IntView initialParticlesPerElem("intial-particle-distribution",Nel);
    IntView cumulativeParticlesOverElem("intial-cumulative-particle-distribution",Nel);
    int adjustedTotParticles = 0;

    Kokkos::parallel_reduce("compute-part-per-elem", Nel, KOKKOS_LAMBDA(const int iel, int& update ){
        double partCount = fracArea(iel)*numParticles;
        initialParticlesPerElem(iel) = (int) partCount + 1;
        update += initialParticlesPerElem(iel);
    },adjustedTotParticles);

    IntView::HostMirror h_initialParticlesPerElem = Kokkos::create_mirror_view(initialParticlesPerElem);
    Kokkos::deep_copy(h_initialParticlesPerElem,initialParticlesPerElem);

    IntView::HostMirror h_cumulativeParticlesOverElem = Kokkos::create_mirror_view(cumulativeParticlesOverElem);
    for (int iel=0; iel<Nel; iel++){
        if (iel == 0){
            h_cumulativeParticlesOverElem(iel) = 0;
        }
        else{
            h_cumulativeParticlesOverElem(iel) = h_cumulativeParticlesOverElem(iel-1) + h_initialParticlesPerElem(iel);
        }
    }
    Kokkos::deep_copy(cumulativeParticlesOverElem,h_cumulativeParticlesOverElem);

    auto rand_pool = RandPool(rngSeed);

    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    Vector2View positions("particle-positions",adjustedTotParticles);
    IntView elementIDs("particle-elementIDs",adjustedTotParticles);
    Kokkos::parallel_for("intialize-particle-positions", Nel, KOKKOS_LAMBDA(const int iel){
        auto rgen = rand_pool.get_state();

        Vector2 v1 = nodes(conn(iel,0));
        Vector2 v2 = nodes(conn(iel,1));
        Vector2 v3 = nodes(conn(iel,2));
        Vector2 v4 = nodes(conn(iel,3));
        int ipartOffset = cumulativeParticlesOverElem(iel);

        for (int ipart=0; ipart < initialParticlesPerElem(iel); ipart++){
            double lambda = Kokkos::rand<RandGen, double>::draw(rgen, 0.0, 1.0);
            double mu = Kokkos::rand<RandGen, double>::draw(rgen, 0.0, 1.0);

            double l1,l2,l3,l4;
            l1 = (1.0-lambda)*(1.0-mu);
            l2 = lambda*(1.0-mu);
            l3 = lambda*mu;
            l4 = (1.0-lambda)*mu;

            Vector2 pos = v1*l1 + v2*l2 + v3*l3 + v4*l4;

            positions(ipart+ipartOffset) = pos;
            elementIDs(ipart+ipartOffset) = iel;
        }

        rand_pool.free_state(rgen);

    });


    Particles partObj(adjustedTotParticles,meshObj,positions,elementIDs);
    return partObj;
}

int Particles::getTotalParticles(){
    return numParticles_;
}

Mesh Particles::getMeshObj(){
    return meshObj_;
}

Vector2View Particles::getParticlePostions(){
    return positions_;
}

IntView Particles::getParticleElementIDs(){
    return elementIDs_;
}

void Particles::validateP2LAlgo(){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    int Nel = meshObj.getTotalElements();
    int numParticles = getTotalParticles();
    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    Kokkos::parallel_for("locate-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
        int iel = eID(ipart);
        bool located = P2LCheck(xp(ipart),
                                nodes(conn(iel,0)),
                                nodes(conn(iel,1)),
                                nodes(conn(iel,2)),
                                nodes(conn(iel,3)));
        if (located)
            printf("ipart=%d -- located\n",ipart);
        else
            printf("NOT LOCATED\n");
    });
}


}
