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
            h_cumulativeParticlesOverElem(iel) = h_cumulativeParticlesOverElem(iel-1) + h_initialParticlesPerElem(iel-1);
        }
    }
    Kokkos::deep_copy(cumulativeParticlesOverElem,h_cumulativeParticlesOverElem);

    auto rand_pool = RandPool(rngSeed);

    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    Vector2View positions("particle-positions",adjustedTotParticles);
    IntView elementIDs("particle-elementIDs",adjustedTotParticles);
    BoolView status("particle-status",adjustedTotParticles);
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
            status(ipart+ipartOffset) = true;
        }

        rand_pool.free_state(rgen);

    });


    Particles partObj(adjustedTotParticles,meshObj,positions,elementIDs,status);
    return partObj;
}

Vector2View getRandDisplacements(int numParticles, int rngSeed, double scaleFactor){
    Vector2View disp("random-displacements",numParticles);
    auto rand_pool = RandPool(rngSeed);
    Kokkos::parallel_for("initialize-random-displacements",numParticles,KOKKOS_LAMBDA(const int ipart){
        auto rgen = rand_pool.get_state();
        double dx = Kokkos::rand<RandGen, double>::draw(rgen, -1.0, 1.0);
        double dy = Kokkos::rand<RandGen, double>::draw(rgen, -1.0, 1.0);
        disp(ipart) = Vector2(dx*scaleFactor,dy*scaleFactor);
        rand_pool.free_state(rgen);
    });
    return disp;
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

BoolView Particles::getParticleStatus(){
    return status_;
}

int Particles::computeTotalActiveParticles(){
    auto status = getParticleStatus();
    int numParticles = getTotalParticles();
    int numActiveParticles = 0;
    Kokkos::parallel_reduce("compute-active-particles",
                            numParticles,
                            KOKKOS_LAMBDA(const int ipart, int &update){
        update += (int) status(ipart);
    },numActiveParticles);

    return numActiveParticles;

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
        int iel = 0;
        bool located = false;
        while (!located && iel<Nel){
            located = P2LCheck(xp(ipart),
                                nodes(conn(iel,0)),
                                nodes(conn(iel,1)),
                                nodes(conn(iel,2)),
                                nodes(conn(iel,3)));
            iel++;
        }

        if (iel-1 != eID(ipart)){
            printf("Particle NOT LOCATED\n");
        }

    });
}

void Particles::T2LTracking(Vector2View dx){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    auto elemFaceBdry = meshObj.getElemFaceBdry();
    int Nel_x = meshObj.getTotalXElements();
    int Nel_y = meshObj.getTotalYElements();
    int numParticles = getTotalParticles();
    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    auto status = getParticleStatus();

    Kokkos::parallel_for("T2L-tracking",numParticles,KOKKOS_LAMBDA(const int ipart){
        int iel = eID(ipart);
        Vector2 xnew = xp(ipart)+dx(ipart);
        bool located = P2LCheck(xnew,
                                nodes(conn(iel,0)),
                                nodes(conn(iel,1)),
                                nodes(conn(iel,2)),
                                nodes(conn(iel,3)));
        FaceDir exitFace = T2LCheckAllFace(xp(ipart),
                                        dx(ipart),
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
        bool inDomain = true;
        bool trackError = false;
        // int iel_x, iel_y;
        while (!located || trackError) {
            // iel_y = iel / Nel_x;
            // iel_x = iel - iel_y*Nel_x;
            // if (iel_x == 0 && exitFace==west){
            //     xnew = findIntersectionPoint(xp(ipart),
            //                                 dx(ipart),
            //                                 nodes(conn(iel,3)),
            //                                 nodes(conn(iel,0)));
            //     located = true;
            //     inDomain = false;
            //     break;
            // }
            // else if (iel_x == Nel_x-1 && exitFace==east){
            //     xnew = findIntersectionPoint(xp(ipart),
            //                                 dx(ipart),
            //                                 nodes(conn(iel,1)),
            //                                 nodes(conn(iel,2)));
            //     located = true;
            //     inDomain = false;
            //     break;
            // }
            // else if (iel_y == 0 && exitFace==south){
            //     xnew = findIntersectionPoint(xp(ipart),
            //                                 dx(ipart),
            //                                 nodes(conn(iel,0)),
            //                                 nodes(conn(iel,1)));
            //     located = true;
            //     inDomain = false;
            //     break;
            // }
            // else if (iel_y == Nel_y-1 && exitFace==north){
            //     xnew = findIntersectionPoint(xp(ipart),
            //                                 dx(ipart),
            //                                 nodes(conn(iel,2)),
            //                                 nodes(conn(iel,3)));
            //     located = true;
            //     inDomain = false;
            //     break;
            // }
            if (elemFaceBdry(iel,exitFace)){
                located = true;
                inDomain = false;
                break;
            }
            switch (exitFace) {
                case east:{
                    iel++;
                    located = P2LCheck(xnew,
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
                    exitFace = T2LCheckForWestEntry(xp(ipart),
                                                dx(ipart),
                                                nodes(conn(iel,0)),
                                                nodes(conn(iel,1)),
                                                nodes(conn(iel,2)),
                                                nodes(conn(iel,3)));
                    break;
                }
                case west:{
                    iel--;
                    located = P2LCheck(xnew,
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
                    exitFace = T2LCheckForEastEntry(xp(ipart),
                                                dx(ipart),
                                                nodes(conn(iel,0)),
                                                nodes(conn(iel,1)),
                                                nodes(conn(iel,2)),
                                                nodes(conn(iel,3)));
                    break;
                }
                case north:{
                    iel += Nel_x;
                    located = P2LCheck(xnew,
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
                    exitFace = T2LCheckForSouthEntry(xp(ipart),
                                                dx(ipart),
                                                nodes(conn(iel,0)),
                                                nodes(conn(iel,1)),
                                                nodes(conn(iel,2)),
                                                nodes(conn(iel,3)));
                    break;
                }
                case south:{
                    iel -= Nel_x;
                    located = P2LCheck(xnew,
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
                    exitFace = T2LCheckForNorthEntry(xp(ipart),
                                                dx(ipart),
                                                nodes(conn(iel,0)),
                                                nodes(conn(iel,1)),
                                                nodes(conn(iel,2)),
                                                nodes(conn(iel,3)));
                    break;
                }
                case none:{
                    printf("no faces crossed -- ERROR\n");
                    trackError = true;
                    break;
                }
            }
        }

        if (!located){
            printf("Particle NOT FOUND\n");
        }
        else{
            eID(ipart) = iel;
            xp(ipart) = xnew;
            status(ipart) = inDomain;
        }
    });
}

// void Particles::MacphersonTracking(Vector2View dx){
//     auto meshObj = getMeshObj();
//     auto nodes = meshObj.getNodesVector();
//     auto conn = meshObj.getConnectivity();
//     int Nel_x = meshObj.getTotalXElements();
//     int Nel_y = meshObj.getTotalYElements();
//     int numParticles = getTotalParticles();
//     auto xp = getParticlePostions();
//     auto eID = getParticleElementIDs();
//     auto status = getParticleStatus();
//
// }

}
