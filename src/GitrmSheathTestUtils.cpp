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

            bool located = P2LCheck(pos,v1,v2,v3,v4);
            if (!located){
                printf("Element ID =%d possibly non-convex -- not passing P2L check for initiated particle\n",iel );
            }
            positions(ipart+ipartOffset) = pos;
            elementIDs(ipart+ipartOffset) = iel;
            status(ipart+ipartOffset) = true;
        }

        rand_pool.free_state(rgen);

    });


    Particles partObj(adjustedTotParticles,meshObj,positions,elementIDs,status);
    return partObj;
}

Particles initializeSingleParticle(Mesh meshObj, unsigned int rngSeed){

    int Nel = meshObj.getTotalElements();
    auto rand_pool = RandPool(rngSeed);

    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    Vector2View positions("particle-positions",1);
    IntView elementIDs("particle-elementIDs",1);
    BoolView status("particle-status",1);
    Kokkos::parallel_for("intialize-particle-position", 1, KOKKOS_LAMBDA(const int){
        auto rgen = rand_pool.get_state();

        double iel_rand = Kokkos::rand<RandGen, double>::draw(rgen, 0.0, 1.0);
        int iel = iel_rand*Nel;

        Vector2 v1 = nodes(conn(iel,0));
        Vector2 v2 = nodes(conn(iel,1));
        Vector2 v3 = nodes(conn(iel,2));
        Vector2 v4 = nodes(conn(iel,3));


        double lambda = Kokkos::rand<RandGen, double>::draw(rgen, 0.0, 1.0);
        double mu = Kokkos::rand<RandGen, double>::draw(rgen, 0.0, 1.0);

        double l1,l2,l3,l4;
        l1 = (1.0-lambda)*(1.0-mu);
        l2 = lambda*(1.0-mu);
        l3 = lambda*mu;
        l4 = (1.0-lambda)*mu;

        Vector2 pos = v1*l1 + v2*l2 + v3*l3 + v4*l4;

        positions(0) = pos;
        elementIDs(0) = iel;
        status(0) = true;

        rand_pool.free_state(rgen);

    });


    Particles partObj(1,meshObj,positions,elementIDs,status);
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
            printf("Particle %d NOT LOCATED with P2L\n",ipart);
        }

    });
}

void Particles::validateP2LAlgoAlt(){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    int Nel = meshObj.getTotalElements();
    int Nel_x = meshObj.getTotalXElements();
    int Nel_y = meshObj.getTotalYElements();
    int numParticles = getTotalParticles();
    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    Kokkos::parallel_for("locate-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
        int iel_x = 0;
        bool located_in_stack = false;
        while (!located_in_stack && iel_x<Nel_x){
            auto node1 = nodes(conn(iel_x,0));
            auto node2 = nodes(conn(iel_x,1));
            int skip = (Nel_y-1)*Nel_x;
            auto node3 = nodes(conn(iel_x+skip,2));
            auto node4 = nodes(conn(iel_x+skip,3));
            located_in_stack = P2LCheck(xp(ipart),
                                        node1,
                                        node2,
                                        node3,
                                        node4);
            iel_x++;
        }

        if (located_in_stack){
            iel_x--;
            int iel_y=0;
            int iel;
            bool located = false;
            while(!located && iel_y<Nel_y){
                iel = iel_x + (iel_y)*Nel_x;
                located = P2LCheck(xp(ipart),
                                    nodes(conn(iel,0)),
                                    nodes(conn(iel,1)),
                                    nodes(conn(iel,2)),
                                    nodes(conn(iel,3)));
                iel_y++;
            }

            if (located && (iel != eID(ipart))){
                printf("Particle %d NOT LOCATED iel=%d eID=%d\n",ipart,iel,eID(ipart));
            }
            // else
                // printf("verified\n");
        }
        else{
            printf("Particle %d NOT IN DOMAIN\n",ipart);
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
        if (status(ipart)){
            int iel = eID(ipart);
            Vector2 xnew = xp(ipart)+dx(ipart);
            FaceDir exitFace = T2LCheck(xp(ipart),
                                        dx(ipart),
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));
            bool inDomain = true;
            while (exitFace != none){
                if (elemFaceBdry(iel,exitFace)){
                    if (exitFace==west){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,3)),
                                                    nodes(conn(iel,0)));
                    }
                    if (exitFace==east){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,1)),
                                                    nodes(conn(iel,2)));
                    }
                    if (exitFace==south){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,0)),
                                                    nodes(conn(iel,1)));
                    }
                    if (exitFace==north){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,2)),
                                                    nodes(conn(iel,3)));
                    }
                    break;
                }

                switch (exitFace) {
                    case east:{
                        iel++;
                        break;
                    }
                    case west:{
                        iel--;
                        break;
                    }
                    case north:{
                        iel += Nel_x;
                        break;
                    }
                    case south:{
                        iel -= Nel_x;
                        break;
                    }
                    case none:{
                        break;
                    }
                }
                exitFace = T2LCheck(xp(ipart),
                                    dx(ipart),
                                    nodes(conn(iel,0)),
                                    nodes(conn(iel,1)),
                                    nodes(conn(iel,2)),
                                    nodes(conn(iel,3)));
            }
            eID(ipart) = iel;
            xp(ipart) = xnew;
            status(ipart) = inDomain;
        }
    });
}

void Particles::T2LTrackingDebug(Vector2View dx){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    auto elemFaceBdry = meshObj.getElemFaceBdry();
    int Nel_x = meshObj.getTotalXElements();
    int Nel_y = meshObj.getTotalYElements();

    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    auto status = getParticleStatus();

    Kokkos::parallel_for("T2L-tracking",1,KOKKOS_LAMBDA(const int ipart){
        if (status(ipart)){
            int iel = eID(ipart);
            printf("xp=(%2.2f,%2.2f)\n",xp(ipart)[0],xp(ipart)[1]);
            printf("dx=(%2.2f,%2.2f)\n",dx(ipart)[0],dx(ipart)[1]);
            Vector2 xnew = xp(ipart)+dx(ipart);
            printf("xnew=(%2.2f,%2.2f)\n",xnew[0],xnew[1]);
            FaceDir exitFace = T2LCheck(xp(ipart),
                                        dx(ipart),
                                        nodes(conn(iel,0)),
                                        nodes(conn(iel,1)),
                                        nodes(conn(iel,2)),
                                        nodes(conn(iel,3)));

            if (exitFace==east)
                printf("Exited thru east face of iel=%d\n",iel );
            if (exitFace==west)
                printf("Exited thru west face of iel=%d\n",iel );
            if (exitFace==south)
                printf("Exited thru south face of iel=%d\n",iel );
            if (exitFace==north)
                printf("Exited thru north face of iel=%d\n",iel );

            bool inDomain = true;

            while (exitFace != none) {
                if (elemFaceBdry(iel,exitFace)){
                    if (exitFace==west){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,3)),
                                                    nodes(conn(iel,0)));
                    }
                    if (exitFace==east){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,1)),
                                                    nodes(conn(iel,2)));
                    }
                    if (exitFace==south){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,0)),
                                                    nodes(conn(iel,1)));
                    }
                    if (exitFace==north){
                        exitFace = none;
                        inDomain = false;
                        xnew = findIntersectionPoint(xp(ipart),
                                                    dx(ipart),
                                                    nodes(conn(iel,2)),
                                                    nodes(conn(iel,3)));
                    }
                    break;
                }

                switch (exitFace) {
                    case east:{
                        iel++;
                        printf("checking in iel=%d\n",iel );
                        exitFace = T2LCheck(xp(ipart),
                                            dx(ipart),
                                            nodes(conn(iel,0)),
                                            nodes(conn(iel,1)),
                                            nodes(conn(iel,2)),
                                            nodes(conn(iel,3)));
                        if (exitFace==east)
                            printf("Exited thru east face of iel=%d\n",iel );
                        if (exitFace==west)
                            printf("Exited thru west face of iel=%d\n",iel );
                        if (exitFace==south)
                            printf("Exited thru south face of iel=%d\n",iel );
                        if (exitFace==north)
                            printf("Exited thru north face of iel=%d\n",iel );
                    break;
                    }
                    case west:{
                        iel--;
                        exitFace = T2LCheck(xp(ipart),
                                            dx(ipart),
                                            nodes(conn(iel,0)),
                                            nodes(conn(iel,1)),
                                            nodes(conn(iel,2)),
                                            nodes(conn(iel,3)));
                        if (exitFace==east)
                            printf("Exited thru east face of iel=%d\n",iel );
                        if (exitFace==west)
                            printf("Exited thru west face of iel=%d\n",iel );
                        if (exitFace==south)
                            printf("Exited thru south face of iel=%d\n",iel );
                        if (exitFace==north)
                            printf("Exited thru north face of iel=%d\n",iel );
                        break;
                    }
                    case north:{
                        iel += Nel_x;
                        exitFace = T2LCheck(xp(ipart),
                                            dx(ipart),
                                            nodes(conn(iel,0)),
                                            nodes(conn(iel,1)),
                                            nodes(conn(iel,2)),
                                            nodes(conn(iel,3)));
                        if (exitFace==east)
                            printf("Exited thru east face of iel=%d\n",iel );
                        if (exitFace==west)
                            printf("Exited thru west face of iel=%d\n",iel );
                        if (exitFace==south)
                            printf("Exited thru south face of iel=%d\n",iel );
                        if (exitFace==north)
                            printf("Exited thru north face of iel=%d\n",iel );

                        break;
                    }
                    case south:{
                        iel -= Nel_x;
                        exitFace = T2LCheck(xp(ipart),
                                            dx(ipart),
                                            nodes(conn(iel,0)),
                                            nodes(conn(iel,1)),
                                            nodes(conn(iel,2)),
                                            nodes(conn(iel,3)));
                        if (exitFace==east)
                            printf("Exited thru east face of iel=%d\n",iel );
                        if (exitFace==west)
                            printf("Exited thru west face of iel=%d\n",iel );
                        if (exitFace==south)
                            printf("Exited thru south face of iel=%d\n",iel );
                        if (exitFace==north)
                            printf("Exited thru north face of iel=%d\n",iel );
                        break;
                    }
                    case none:{
                        break;
                    }
                }
            }

            eID(ipart) = iel;
            xp(ipart) = xnew;
            status(ipart) = inDomain;
        }
    });
}


void Particles::interpolateQuadEField(){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    auto Efield = meshObj.getEfieldVector();
    auto elemFaceBdry = meshObj.getElemFaceBdry();
    int Nel_x = meshObj.getTotalXElements();
    int Nel_y = meshObj.getTotalYElements();
    int numParticles = getTotalParticles();
    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    auto status = getParticleStatus();

    Kokkos::parallel_for("Efield-2-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
        if (status(ipart)){
            int iel = eID(ipart);
            double lambda, mu;
            getCoeffsForQuadBC(xp(ipart),
                            nodes(conn(iel,0)),
                            nodes(conn(iel,1)),
                            nodes(conn(iel,2)),
                            nodes(conn(iel,3)),
                            &lambda,
                            &mu);
            Vector2 Ep = Efield(conn(iel,0))*(1.0-lambda)*(1.0-mu) +
                         Efield(conn(iel,1))*lambda*(1.0-mu) +
                         Efield(conn(iel,2))*lambda*mu +
                         Efield(conn(iel,3))*(1.0-lambda)*mu;
        }
    });

}

void Particles::interpolateTriEField(){
    auto meshObj = getMeshObj();
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    auto Efield = meshObj.getEfieldVector();
    auto elemFaceBdry = meshObj.getElemFaceBdry();
    int Nel_x = meshObj.getTotalXElements();
    int Nel_y = meshObj.getTotalYElements();
    int numParticles = getTotalParticles();
    auto xp = getParticlePostions();
    auto eID = getParticleElementIDs();
    auto status = getParticleStatus();

    Kokkos::parallel_for("Efield-2-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
        if (status(ipart)){
            int iel = eID(ipart);
            double lambda0, lambda1, lambda2, lambda3;
            getTriangleBC(xp(ipart),
                            nodes(conn(iel,0)),
                            nodes(conn(iel,1)),
                            nodes(conn(iel,2)),
                            nodes(conn(iel,3)),
                            &lambda0, &lambda1,
                            &lambda2, &lambda3);
            Vector2 Ep = Efield(conn(iel,0))*lambda0 +
                         Efield(conn(iel,1))*lambda1 +
                         Efield(conn(iel,2))*lambda2 +
                         Efield(conn(iel,3))*lambda3;

            // Vector2 xp_computed = nodes(conn(iel,0))*lambda0 +
            //              nodes(conn(iel,1))*lambda1 +
            //              nodes(conn(iel,2))*lambda2 +
            //              nodes(conn(iel,3))*lambda3;
            //
            // Vector2 diff = xp_computed-xp(ipart);
            // printf("%2.5e %2.5e\n",diff[0],diff[1]);
            xp(ipart) += Ep*0.0;
        }
    });
}


void Particles::interpolateWachpress(){

auto meshObj = getMeshObj();
auto nodes = meshObj.getNodesVector();
auto conn = meshObj.getConnectivity();
auto Efield = meshObj.getEfieldVector();
auto elemFaceBdry = meshObj.getElemFaceBdry();
int Nel_x = meshObj.getTotalXElements();
int Nel_y = meshObj.getTotalYElements();
int numParticles = getTotalParticles();
auto xp = getParticlePostions();
auto eID = getParticleElementIDs();
auto status = getParticleStatus();


Kokkos::parallel_for("Efield-2-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
if (status(ipart)){
int iel = eID(ipart);
double w1,w2,w3,w4;
auto v1 = nodes(conn(iel,0));
auto v2 = nodes(conn(iel,1));
auto v3 = nodes(conn(iel,2));
auto v4 = nodes(conn(iel,3));

printf("input particle coordinate:\n (%1.3e,%1.3e)\n",xp(ipart)[0],xp(ipart)[1]);

getWachpressCoeffs(xp(ipart),v1,v2,v3,v4,&w1,&w2,&w3,&w4);

auto wp_coord = v1*w1+v2*w2+v3*w3+v4*w4;


printf("coordinate from Wachspress interpolation:\n (%1.3e,%1.3e)\n",wp_coord[0],wp_coord[1]);

double w[maxVerti] = {0.0};// all init to 0.0 can 
Vector2 v[maxVerti+1];
int numEverts = 4;
for(int i = 0; i<numEverts; i++){
    v[i] = nodes(conn(iel,i));
}
v[numEverts] = nodes(conn(iel,0));
getWachpressCoeffs(xp(ipart), numEverts, v, w);
Vector2 wp_coord2(0,0);

printf("Test\n(%1.3e,%1.3e)\n",v[6][0],v[6][1]);
for(int i = 0; i<numEverts; i++){
	wp_coord2 = wp_coord2 + v[i]*w[i]; 
}

printf("(%1.3e,%1.3e)\n",wp_coord2[0],wp_coord2[1]);
wp_coord2 = wp_coord2 + v[6]*w[6];
printf("(%1.3e,%1.3e)\n",wp_coord2[0],wp_coord2[1]);
printf("coordinate from new Wachspress interpolation:\n (%1.3e,%1.3e)\n",wp_coord2[0],wp_coord2[1]);
printf("coordinate difference between two Wachspress interpolation:\n (%1.3e,%1.3e)\n",wp_coord[0]-wp_coord2[0],wp_coord[1]-wp_coord2[1]);

//test zero weight

}
														       });

}

}
