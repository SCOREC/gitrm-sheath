/*!
* \author Vignesh Vittal-Srinivasaragavan
* \date 05-19-2022
* \mainpage Sheath Electric Field Model for GITRm (with Kokkos)
*/

#ifndef GitrmSheath_hpp
#define GitrmSheath_hpp

#include <iostream>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>

#include <Kokkos_Core.hpp>
#include "GitrmSheathUtils.hpp"

namespace sheath {

using Vector2View = Kokkos::View<Vector2*>;
using Int4View = Kokkos::View<int*[4]>;

class Mesh{
private:
    int Nel_x_;
    int Nel_y_;
    Vector2View nodes_;
    Int4View conn_;

public:
    Mesh(){};

    Mesh(int Nel_x,
         int Nel_y,
         Vector2View nodes,
         Int4View conn):
         Nel_x_(Nel_x),
         Nel_y_(Nel_y),
         nodes_(nodes),
         conn_(conn){};

    int getTotalNodes();
    int getTotalElements();
    Vector2View getNodesVector();
    Int4View getConnectivity();

};

Mesh initializeSheathMesh(int Nel_x,
                          int Nel_y,
                          std::string coord_file);

} // namespace sheath

#endif
