#include "GitrmSheath.hpp"

namespace sheath {

int Mesh::getTotalNodes(){
    return nnpTotal_;
}

int Mesh::getTotalElements(){
    return nelTotal_;
}

int Mesh::getTotalXElements(){
    return Nel_x_;
}

int Mesh::getTotalYElements(){
    return Nel_y_;
}

Vector2View Mesh::getNodesVector(){
    return nodes_;
}

Vector2View Mesh::getEfieldVector(){
    return Efield_;
}

Int4View Mesh::getConnectivity(){
    return conn_;
}

Int4View Mesh::getElemFaceBdry(){
    return elemFaceBdry_;
}

Mesh initializeSimpleMesh(){
   int Nel = 1;
   int Nnp = 4;

   Vector2View node("node-coord-vector",Nnp);
   Vector2View::HostMirror h_node = Kokkos::create_mirror_view(node);

   h_node(0) = Vector2(0.63,-0.1);
   h_node(1) = Vector2(20.3,0.2);
   h_node(2) = Vector2(23.7,3.12);
   h_node(3) = Vector2(-2.1,2.04);

   Kokkos::deep_copy(node, h_node);
   
   Vector2View Efield("Efield-vector",Nnp); 
   //Vector2View::HostMirror h_Efield = Kokkos::create_mirror_view(Efield);

   //h_Efield(0) = Vector2(10.0,-10.0);
   //h_Efield(1) = Vector2(20.0,-20.0);
   //h_Efield(2) = Vector2(15.0,-15.0);
   //h_Efield(3) = Vector2(0.0,-0.0);

   //Kokkos::deep_copy(Efield, h_Efield);

   Int4View conn("elem-connectivty",Nel);
   Int4View elemFaceBdry("elem-face-boundary",Nel);

   Int4View::HostMirror h_conn = Kokkos::create_mirror_view(conn);
   h_conn(0,0) = 0;
   h_conn(0,1) = 1;
   h_conn(0,2) = 2;
   h_conn(0,3) = 3;
   Kokkos::deep_copy(conn, h_conn);

   return Mesh(1,1,node,conn,elemFaceBdry,Nel,Nnp,Efield);
}

Mesh initializeTestMesh(int factor){
    int Nel_size = 11; 
    int Nnp_size = 15;
    int Nel = Nel_size*factor;
    int Nnp = Nnp_size*factor;
    double v_array[15][2] = {{0,0},{0.49,0},{1,0},{0.6,0.25},{0,0.5},{0.2,0.6},{0.31,0.6},{0.45,0.55},{0.6,0.4},{0.7,0.6},{1,0.54},{0.25,0.77},{0,1},{0.47,1},{1,1}};
    int conn_array[11][8] = {{1,2,4,9,8,7,6,5},{2,3,4,-1,-1,-1,-1,-1},{4,3,9,-1,-1,-1,-1,-1},{9,3,11,10,-1,-1,-1,-1},{10,11,15,14,-1,-1,-1,-1},{8,9,10,14,-1,-1,-1,-1},{7,8,14,12,-1,-1,-1,-1},{13,12,14,-1,-1,-1,-1,-1},{6,7,12,-1,-1,-1,-1,-1},{6,12,13,-1,-1,-1,-1,-1},{5,6,13,-1,-1,-1,-1,-1}};
    int node_array[11] = {8,3,3,4,4,4,4,3,3,3,3};
    Vector2View node("node-coord-vector", Nnp);
    Vector2View::HostMirror h_node = Kokkos::create_mirror_view(node);
    for(int j=0; j<factor; j++){
        for(int i=0; i<Nnp_size; i++){
            h_node(i+j*Nnp_size) = Vector2(v_array[i][0],v_array[i][1]); 
        }
    }
    Kokkos::deep_copy(node, h_node);

    Vector2View Efield("Efield-vector",Nnp); 
    Int4View conn("elem-connectivty",Nel);
    Int4View elemFaceBdry("elem-face-boundary",Nel);

    Int4View::HostMirror h_conn = Kokkos::create_mirror_view(conn);
    for(int f=0; f<factor; f++){
        for(int i=0; i<Nel_size; i++){
            h_conn(i+f*Nel_size,0) = node_array[i];
            for(int j=0; j<h_conn(i,0); j++){
                h_conn(i+f*Nel_size,j+1) = conn_array[i][j] + f*Nnp_size;
            }
        }
    }
    Kokkos::deep_copy(conn, h_conn);

    return Mesh(1,1,node,conn,elemFaceBdry,Nel,Nnp,Efield);   
}

Mesh initializeSheathMesh(int Nel_x,
                          int Nel_y,
                          std::string coord_file,
                          std::string Efield_file){

    int Nel = Nel_x * Nel_y;
    int Nnp = (Nel_x+1) * (Nel_y+1);
    
    Vector2View node("node-coord-vector",Nnp);

    Vector2View::HostMirror h_node = Kokkos::create_mirror_view(node);

    std::ifstream coordFile(coord_file);

    double xnode, ynode;
    int inp = 0;
    if (coordFile.is_open()){
        while (coordFile >> xnode >> ynode){
            if (inp >= Nnp){
                std::cout << "ERROR: Too many nodes in input file -- re-check inputs\n";
                exit(0);
            }
            h_node(inp) = Vector2(xnode,ynode);
            inp++;
        }
    }
    else{
        std::cout << "ERROR: Node coordinate file " << coord_file << " INVALID\n";
        exit(0);
    }
    coordFile.close();

    if (inp < Nnp){
        std::cout << "ERROR: Insufficient nodes in node input file -- re-check inputs\n";
        exit(0);
    }

    Kokkos::deep_copy(node, h_node);


    Vector2View Efield("Efield-vector",Nnp);

    Vector2View::HostMirror h_Efield = Kokkos::create_mirror_view(Efield);
    std::ifstream EfieldFile(Efield_file);

    double Ex, Ey;
    inp = 0;
    if (EfieldFile.is_open()){
        while (EfieldFile >> Ex >> Ey){
            if (inp >= Nnp){
                std::cout << "ERROR: Too many nodes in Efield input file -- re-check inputs\n";
                exit(0);
            }
            h_Efield(inp) = Vector2(Ex,Ey);
            inp++;
        }
    }
    else{
        std::cout << "ERROR: Efield file " << Efield_file << " INVALID\n";
        exit(0);
    }
    EfieldFile.close();

    Kokkos::deep_copy(Efield, h_Efield);

    Int4View conn("elem-connectivty",Nel);
    Int4View elemFaceBdry("elem-face-boundary",Nel);
    Kokkos::parallel_for("init-elem-connectivty", Nel, KOKKOS_LAMBDA(const int iel){
        int iel_y = iel / Nel_x;
        int iel_x = iel - iel_y*Nel_x;
        conn(iel,0) = iel + iel_y;
        conn(iel,1) = conn(iel,0)+1;
        conn(iel,2) = conn(iel,1)+1+Nel_x;
        conn(iel,3) = conn(iel,2)-1;
        elemFaceBdry(iel,0) = 0;
        elemFaceBdry(iel,1) = 0;
        elemFaceBdry(iel,2) = 0;
        elemFaceBdry(iel,3) = 0;
        if (iel_x == 0){
            elemFaceBdry(iel,3) = 1;
        }
        if (iel_x == Nel_x-1){
            elemFaceBdry(iel,1) = 1;
        }
        if (iel_y == 0){
            elemFaceBdry(iel,0) = 1;
        }
        if (iel_y == Nel_y-1){
            elemFaceBdry(iel,2) = 1;
        }

    });

    return Mesh(Nel_x,Nel_y,node,conn,elemFaceBdry,Nel,Nnp,Efield);
}

void Mesh::computeFractionalElementArea(){
    int Nel = getTotalElements();

    double totArea = 0.0;
    auto conn = conn_;
    auto nodes = nodes_;
    DoubleView fracArea("fractional-areas",Nel);
    Kokkos::parallel_reduce("computing-elem-areas",
                            Nel,
                            KOKKOS_LAMBDA (const int iel, double& update ){

    Vector2 p1 = nodes(conn(iel,0));
    Vector2 p2 = nodes(conn(iel,1));
    Vector2 p3 = nodes(conn(iel,2));
    Vector2 p4 = nodes(conn(iel,3));

    // printf("(%2.5e %2.5e) (%2.5e %2.5e) (%2.5e %2.5e) (%2.5e %2.5e)\n",
    //                         nodes(conn(iel,0))[0],nodes(conn(iel,0))[1],
    //                         nodes(conn(iel,1))[0],nodes(conn(iel,1))[1],
    //                         nodes(conn(iel,2))[0],nodes(conn(iel,2))[1],
    //                         nodes(conn(iel,3))[0],nodes(conn(iel,3))[1]);

    Vector2 e1 = p2-p1;
    Vector2 e2 = p4-p1;
    Vector2 diag = p3-p1;

    double baseTri = diag.magnitude();

    double lambdaLowerTri = diag.dot(e1)/(baseTri*baseTri);
    double lambdaUpperTri = diag.dot(e2)/(baseTri*baseTri);

    Vector2 hLowerTri = e1-diag*lambdaLowerTri;
    Vector2 hUpperTri = e2-diag*lambdaUpperTri;

    double heightLowerTri = hLowerTri.magnitude();
    double heightUpperTri = hUpperTri.magnitude();

    fracArea(iel) = 0.5*baseTri*(heightLowerTri+heightUpperTri);
    update += fracArea(iel);

    }, totArea);

    totArea_ = totArea;

    Kokkos::parallel_for("computing-fractional-areas",
                         Nel,
                         KOKKOS_LAMBDA(const int iel){
        fracArea(iel) /= totArea;
    });

    fracArea_ = fracArea;
}

double Mesh::getTotalArea(){
    return totArea_;
}

DoubleView Mesh::getFractionalElementAreas(){
    return fracArea_;
}

} // namespace sheath
