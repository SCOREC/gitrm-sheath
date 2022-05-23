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

Int4View Mesh::getConnectivity(){
    return conn_;
}

Int4View Mesh::getElemFaceBdry(){
    return elemFaceBdry_;
}

Mesh initializeSheathMesh(int Nel_x,
                          int Nel_y,
                          std::string coord_file){

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
        std::cout << "ERROR: Insufficient nodes in input file -- re-check inputs\n";
        exit(0);
    }

    Kokkos::deep_copy(node, h_node);

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

    return Mesh(Nel_x,Nel_y,node,conn,elemFaceBdry,Nel,Nnp);
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
