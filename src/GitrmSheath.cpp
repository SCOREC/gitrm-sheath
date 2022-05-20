#include "GitrmSheath.hpp"

namespace sheath {

int Mesh::getTotalNodes(){
    int Nnp = nodes_.extent(0);
    return Nnp;
}

int Mesh::getTotalElements(){
    int Nnp = conn_.extent(0);
    return Nnp;
}

Vector2View Mesh::getNodesVector(){
    return nodes_;
}

Int4View Mesh::getConnectivity(){
    return conn_;
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

    Kokkos::View<int*[4]> conn("elem-connectivty",Nel);
    Kokkos::parallel_for("init-elem-connectivty", Nel, KOKKOS_LAMBDA(const int iel){
        int iel_y = iel / Nel_x;
        int iel_x = iel - iel_y*Nel_x;
        conn(iel,1) = iel + iel_x;
        conn(iel,2) = conn(iel,1)+1;
        conn(iel,3) = conn(iel,2)+1+Nel_x;
        conn(iel,4) = conn(iel,3)-1;
    });

    return Mesh(Nel_x,Nel_y,node,conn);
}

} // namespace sheath
