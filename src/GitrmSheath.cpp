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

IntView Mesh::getElem2Particles(){
    return elem2Particles_;
}

IntElemsPerVertView Mesh::getVertex2Elems(){
    return vertex2Elems_;
}

void Mesh::setElem2Particles(IntView elem2Particles){
    elem2Particles_ = IntView("elementToParticles",elem2Particles.size());
    Kokkos::deep_copy(elem2Particles_, elem2Particles);
}

void Mesh::setVertex2Elems(IntElemsPerVertView vertex2Elems){
    vertex2Elems_ = IntElemsPerVertView("vertexToElements",vertex2Elems.size());
    Kokkos::deep_copy(vertex2Elems_,vertex2Elems);
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

    //creat a new mesh
    //random num of particles (4-6)
    //timing
    //clean up and presentable
    //assembly ?
    //particle tracking
    int Nel_size = 10;//10 tri-oct
    int Nnp_size = 19;//arbi <20
    int Nel = Nel_size*factor;
    int Nnp = Nnp_size*factor;
    double v_array[19][2] = {{0.00,0.00},{0.47,0.00},{1.00,0.00},{0.60,0.25},{0.60,0.40},{0.00,0.50},{0.31,0.60},{0.40,0.60},{0.45,0.55},{0.70,0.49},{0.80,0.45},{0.90,0.47},{1.00,0.55},{0.60,0.60},{0.37,0.80},{0.00,1.00},{0.37,1.00},{0.48,1.00},{1.00,1.00}};
    int conn_array[10][8] = {{1,2,4,5,9,8,7,6},{2,3,4,-1,-1,-1,-1,-1},{4,3,11,10,5,-1,-1,-1},
    {3,12,11,-1,-1,-1,-1,-1},{3,13,12,-1,-1,-1,-1,-1},
    {5,10,14,9,-1,-1,-1,-1},{9,14,18,17,15,8,-1,-1},
    {7,8,15,-1,-1,-1,-1,-1},{6,7,15,17,16,-1,-1,-1},
    {14,10,11,12,13,19,18,-1}};
    int node_array[10] = {8,3,5,3,3,4,6,3,5,7};
    int vertex2Elems_array[19][6] = {{1,0,-1,-1,-1,-1},{2,0,1,-1,-1,-1},{4,1,2,3,4,-1},{3,0,1,2,-1,-1},{3,0,2,5,-1,-1},{2,0,8,-1,-1,-1},{3,0,7,8,-1,-1},{3,0,6,7,-1,-1},{3,0,5,6,-1,-1},{3,2,5,9,-1,-1},{3,2,3,9,-1,-1},{3,3,4,9,-1,-1},{2,4,9,-1,-1,-1},{3,5,6,9,-1,-1},{3,6,7,8,-1,-1},{1,8,-1,-1,-1,-1},{2,6,8,-1,-1,-1},{2,6,9,-1,-1,-1},{1,9,-1,-1,-1,-1}};
    
    Vector2View node("node-coord-vector", Nnp);
    IntElemsPerVertView vertex2Elems("vertexToElements",Nnp);
    
    Vector2View::HostMirror h_node = Kokkos::create_mirror_view(node);
    IntElemsPerVertView::HostMirror h_vertex2Elems = Kokkos::create_mirror_view(vertex2Elems);
    for(int f=0; f<factor; f++){
        for(int i=0; i<Nnp_size; i++){
            h_node(i+f*Nnp_size) = Vector2(v_array[i][0],v_array[i][1]); 
            h_vertex2Elems(i+f*Nnp_size,0) = vertex2Elems_array[i][0];
            for(int j=1; j<=vertex2Elems_array[i][0]; j++){
                h_vertex2Elems(i+f*Nnp_size,j) = vertex2Elems_array[i][j]+f*Nel_size; 
            }
        }
    }
    Kokkos::deep_copy(node, h_node);
    Kokkos::deep_copy(vertex2Elems,h_vertex2Elems);
    /*
    //print the 13th(last) unit and check
    for(int i=Nnp_size*(factor-1); i<Nnp; i++){
        printf("%d: ",i);
        for(int j=0;j<=h_vertex2Elems(i,0);j++){
            printf("%d ",h_vertex2Elems(i,j));   
        }
        printf("\n");
    }
    //*/
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
    
    IntView elem2Particles("notInitElem2Particles",0);
    return Mesh(1,1,node,conn,elemFaceBdry,Nel,Nnp,Efield,elem2Particles,vertex2Elems);   

Mesh readMPASMesh(int& ncid){
    int retval,
        nCells, nCellsID,
        nVertices, nVerticesID,
        maxEdges, maxEdgesID;
    size_t temp;

    int xVertexID, yVertexID, zVertexID, verticesOnCellID, cellsOnVertexID;
    double* xVertex, yVertex, zVertex; //nVertices
    int** verticesOnCell;     //[maxEdges,nCells]
    double** cellsOnVertex;   //[3,nVertices]
    if ((retval = nc_inq_dimid(ncid, "nCells", &nCellsID)))
        ERRexit(retval);
    if ((retval = nc_inq_dimid(ncid, "nVertices", &nVerticesID)))
        ERRexit(retval);
    if ((retval = nc_inq_dimid(ncid, "maxEdges", &maxEdgesID)))
        ERRexit(retval);
    
    
    if ((retval = nc_inq_dimlen(ncid, nCellsID, &temp)))
        ERRexit(retval);
    nCells = temp;
    if ((retval = nc_inq_dimlen(ncid, nVerticesID, &temp)))
        ERRexit(retval);
    nVertices = temp;
    if ((retval = nc_inq_dimlen(ncid, maxEdgesID, &temp)))
        ERRexit(retval);
    maxEdges = temp;

    
    if ((retval = nc_inq_varid(ncid, "xVertex", &xVertexID)))
        ERRexit(retval);
    if ((retval = nc_inq_varid(ncid, "yVertex", &yVertexID)))
        ERRexit(retval);
    if ((retval = nc_inq_varid(ncid, "zVertex", &zVertexID)))
        ERRexit(retval);
    if ((retval = nc_inq_varid(ncid, "verticesOnCell", &verticesOnCellID)))
        ERRexit(retval);
    if ((retval = nc_inq_varid(ncid, "celsOnVertex", &cellsOnVertexID)))
        ERRexit(retval);

    xVertex = new double[nVertices];
    yVertex = new double[nVertices];
    zVertex = new double[nVertices];
    verticesOnCell = new int*[maxEdges];
    for(int i=0; i<maxEdges; i++)
        verticesOnCell[i] = new int[nCells];
    cellsOnVertex = new double*[3]; //vertex dimension is 3
    for(int i=0; i<3; i++)
        cellsOnVertex = new double[];

    
    if ((retval = nc_get_var(ncid, xVertexID, &xVertex)))
        ERRexit(retval);
    if ((retval = nc_get_var(ncid, yVertexID, &yVertex)))
        ERRexit(retval);
    if ((retval = nc_get_var(ncid, zVertexID, &zVertex)))
        ERRexit(retval);
    if ((retval = nc_get_var(ncid, xVertexID, &xVertex)))
        ERRexit(retval);
    if ((retval = nc_get_var(ncid, xVertexID, &xVertex)))
        ERRexit(retval);
    //delete dynamic allocation
    delete [] xVertex;
    delete [] yVertex;
    delete [] zVertex;
    for(int i=0; i<maxEdges; i++)
        delete [] verticesOnCell[i];
    delete [] verticesOnCell;
    for(int i=0; i<3; i++)
        delete [] cellsOnVertex[i];
    delete [] cellsOnVertex;


    
    //Vector2View   
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
