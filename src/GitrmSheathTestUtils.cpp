#include "GitrmSheathTestUtils.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Atomic.hpp>

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

Particles initializeTestParticles(Mesh meshObj){

    int Nel = meshObj.getTotalElements();
    int Nnp = meshObj.getTotalNodes();

    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
   ///*
    int numPart = 0; //need to do this on host
    //  numPart++ by num vertices for elem -parallel reduction
    //Kokkos::parallel_reduce("sum_numVerti",Nel,KOKKOS_LAMBDA(const int& i, int& sum){
    //    sum += conn(i,0);
        //printf("%d\n", conn(i,0));
    //},numPart);
    //printf("%d-%d-%d\n",numPart,Nel,Nnp);
    /* calc the positions once
    Vector2View position("particle-positions",11);
    Kokkos::parallel_for("initialize-positions",11, KOKKOS_LAMBDA(const int iel){
        int numConn = conn(iel,0);
        double sum_x = 0.0, sum_y = 0.0;
        for(int i=1; i<=numConn; i++){
            sum_x += nodes(conn(iel,i)-1)[0];
            sum_y += nodes(conn(iel,i)-1)[1];
        }
        position(iel) = Vector2(sum_x/numConn, sum_y/numConn);
    });
    //*/
    //  offset array -parallel scan
    IntView numParticlesPerElement("numParticlesPerElement",Nel);
    Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
    Kokkos::parallel_for("setnumParticles",Nel, KOKKOS_LAMBDA(const int i){
        auto generator = random_pool.get_state();
        numParticlesPerElement(i) = generator.urand(4,7);   
        random_pool.free_state(generator);
        //numParticlesPerElement(i) = 6;
    });
    Kokkos::fence();
    
    Kokkos::parallel_reduce("totalParticles",Nel,KOKKOS_LAMBDA(const int&i, int& sum){
        sum += numParticlesPerElement(i);
    },numPart);
    IntView particleToElement("particleToElement",numPart);

    IntView elem2Particles("elementToParticles",Nel*(maxParts+1));//maxParts=8
    Kokkos::parallel_scan("setParticleToElement", Nel, KOKKOS_LAMBDA(int i, int& ipart, bool is_final){
        //do the elem2Particles
        if(is_final){  
            elem2Particles(i*(maxParts+1)) = numParticlesPerElement(i);
            for(int j=0; j<numParticlesPerElement(i); j++){
                particleToElement(ipart+j) = i;
                elem2Particles(i*(maxParts+1)+j+1) = ipart+j;
            }   
        }
        ipart += numParticlesPerElement(i); 
    },numPart);
    //printf("numPart: %d,Nel*6: %d\n",numPart,Nel*6);
    //Kokkos::parallel_for("testForE2P", Nel*(maxParts+1), KOKKOS_LAMBDA(const int i){
    //   printf("%d:(%d)\n",i,elem2Particles(i));
    //});
    Vector2View positions("particle-positions",numPart);
    IntView elementIDs("particle-elementIDs",numPart);
    BoolView status("particle-status",numPart);

    Kokkos::parallel_for("intialize-particle-position", numPart, KOKKOS_LAMBDA(const int iPart){
        int iel = particleToElement(iPart);
        int numVerti = conn(iel,0);
        double sum_x = 0.0, sum_y = 0.0;
        for(int i=1; i<= numVerti; i++){
            sum_x += nodes(conn(iel,i)-1)[0];
            sum_y += nodes(conn(iel,i)-1)[1];
        }
        positions(iPart) = Vector2(sum_x/numVerti, sum_y/numVerti);
        elementIDs(iPart) = iel;
        status(iPart) = true;
    });
    
    //Kokkos::parallel_for("test-check", numPart, KOKKOS_LAMBDA(const int iPart){
    //    printf("%d: %.3f,%.3f\n",iPart, positions(iPart)[0],positions(iPart)[1]);
    //});
    //Mesh temp(1,1,nodes,offset_conn,meshObj.getElemFaceBdry(),Nel,Nnp,meshObj.getEfieldVector());
    meshObj.setElem2Particles(elem2Particles);
    //printf("after set:%ld\n",meshObj.getElem2Particles().size());
    return Particles(numPart, meshObj, positions, elementIDs, status);    
}

void assembly(Mesh meshObj, Particles ptclObj){
    meshObj = ptclObj.getMeshObj();
    int Nel = meshObj.getTotalElements();
    int Nnp = meshObj.getTotalNodes();

    int numParti = ptclObj.getTotalParticles();
    //auto eID = partObj.getParticleElementIDs();   
    
    auto nodes = meshObj.getNodesVector();
    auto conn = meshObj.getConnectivity();
    auto elem2Particles = meshObj.getElem2Particles();
    auto xp = ptclObj.getParticlePostions();

    DoubleView vField("vField",Nnp);
    
    Kokkos::parallel_for("vertex_assem",Nel, KOKKOS_LAMBDA(const int iel){
        int nVertsE = conn(iel,0);
        int nParticlesE = elem2Particles(iel*(maxParts+1)); 
        for(int i=0; i<nVertsE; i++){
            int vID = conn(iel,i+1)-1;
            auto vertexLoc = nodes(vID);
            //atomic_increment(&vfield(vID));
            //calc the sum distance to the vertex to the particles
            for(int j=0; j<nParticlesE; j++){
                int partID = elem2Particles(iel*(maxParts+1)+j+1);
                double distance = (xp(partID)-vertexLoc).magnitude();
                Kokkos::atomic_add(&vField(vID),distance);
            }
            //printf("%d\n",/*iel,vID,*/nParticles);
            //add the sum up;
            //Kokkos::atomic_add(&vField(vID),1);
        }
    });
    //ptcl
    auto eID = ptclObj.getParticleElementIDs();
    DoubleView vField2("vField2",Nnp);
    Kokkos::parallel_for("vertex_assem2", numParti, KOKKOS_LAMBDA(const int ipart){
        int iel = eID(ipart); 
        int nVertsE = conn(iel,0);
        for(int i=0; i<nVertsE; i++){
            int vID = conn(iel,i+1)-1;
            auto vertexLoc = nodes(vID);
            //atomic_increment(&vfield(vID));
            //calc the sum distance to the vertex to the particles
            double distance = (xp(ipart)-vertexLoc).magnitude();
            Kokkos::atomic_add(&vField2(vID),distance);
        }
    });
    
    //Kokkos::parallel_for("vFieldcheck",Nnp, KOKKOS_LAMBDA(const int i){
    //    printf("%.3e, %.3e\n",vField(i),vField2(i));
    //});
//*/

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

    //100 1000 10000 100000 1000000 |(mem out)10000000
    numParticles = 1;
    Kokkos::parallel_for("Efield-2-particles",numParticles,KOKKOS_LAMBDA(const int ipart){
        if (status(ipart)){
            int iel = eID(ipart);
            //Vector2 wp_coord(0,0);
            //double w[maxVerti] = {0.0};// all init to 0.0 can 
            Vector2 v[maxVerti+1] = {nodes(conn(iel,1))};
            //std::array<Vector2,maxVerti+1> v;
            initArrayWith(v,maxVerti+1,nodes(conn(iel,1)));
            int numEverts = conn(iel,0);
            for(int i = 1; i<=numEverts; i++){
                v[i-1] = nodes(conn(iel,i)-1);
                printf("%f %f\n",v[i-1][0],v[i-1][1]);
            }
            printf("xp:%f,%f",xp(ipart)[0],xp(ipart)[1]);
            //1 2 ... n 1
            v[numEverts] = nodes(conn(iel,1)-1);

            //getWachpressCoeffs(xp(ipart), numEverts, v, w);
            //if(numEverts != maxVerti){
            //    for(int i = numEverts+1; i<maxVerti+1; i++ ){
            //        v[i] = v[numEverts];
            //    }
            //}
             
            double wByArea[maxVerti] = {0.0};
            //printf("test%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",wByArea[0],wByArea[1],wByArea[2],wByArea[3],wByArea[4],wByArea[5],wByArea[6],wByArea[7]);
            //std::array<double,maxVerti> wByArea;
            initArrayWith(wByArea,maxVerti,0.0);
            getWachpressCoeffsByArea(xp(ipart), numEverts, v, wByArea);
            
            double wByHeight[maxVerti] = {0.0};
            Vector2 gradWByHeight[maxVerti];
            gradient(xp(ipart), numEverts, v, wByHeight, gradWByHeight);
            
            double wMPAS[maxVerti] = {0.0};
            Vector2 gradWMPAS[maxVerti];
            gradientMPAS(xp(ipart), numEverts, v, wMPAS, gradWMPAS);

            Vector2 wp_coordByArea(0,0);
            Vector2 wp_coordByGradient(0,0);

            Vector2 gradFByHeightAtP(0,0);
            Vector2 gradFMPASAtP(0,0);
            for(int i = 0; i<maxVerti; i++){
	        double Fi = 1 + 10.36*v[i][0]+12.2*v[i][1];
                gradFByHeightAtP = Vector2(gradFByHeightAtP[0] + Fi*gradWByHeight[i][0],gradFByHeightAtP[1] + Fi*gradWByHeight[i][1]);
                gradFMPASAtP = Vector2(gradFMPASAtP[0] + Fi*gradWMPAS[i][0],gradFMPASAtP[1] + Fi*gradWMPAS[i][1]);
	        wp_coordByArea = wp_coordByArea + v[i]*wByArea[i]; 
	        wp_coordByGradient = wp_coordByGradient + v[i]*wByHeight[i];  
            }   
                //print AtP[0]  AtP[1]
                //check 10.36   12.2
            printf("gradFByHeightAtP= (%6.3f,%6.3f) |gradFMPAS= (%6.3f,%6.3f)\n",gradFByHeightAtP[0],gradFByHeightAtP[1],gradFMPASAtP[0],gradFMPASAtP[1]);
            
            //if(iel%11 == 0){
            //printf("coordinate from %d interpolation:\n point(%1.3e,%1.3e) wpByArea:(%1.3e,%1.3e) wpByGradient:(%1.3e,%1.3e)\n",ipart,xp(ipart)[0],xp(ipart)[1],wp_coordByArea[0],wp_coordByArea[1],wp_coordByGradient[0],wp_coordByGradient[1]);
            //}

        }
    });
}

}
