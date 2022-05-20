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
#include <Kokkos_Random.hpp>

#include "GitrmSheathUtils.hpp"

namespace sheath {

using Vector2View = Kokkos::View<Vector2*>;
using Int4View = Kokkos::View<int*[4]>;
using DoubleView = Kokkos::View<double*>;
using IntView = Kokkos::View<int*>;

using RandPool = Kokkos::Random_XorShift64_Pool<>;
using RandGen  = RandPool::generator_type;


class Mesh{
private:
    int Nel_x_;
    int Nel_y_;
    Vector2View nodes_;
    Int4View conn_;
    int nnpTotal_;
    int nelTotal_;

    double totArea_;
    DoubleView fracArea_;

public:
    Mesh(){};

    Mesh(int Nel_x,
         int Nel_y,
         Vector2View nodes,
         Int4View conn,
         int nelTotal,
         int nnpTotal):
         Nel_x_(Nel_x),
         Nel_y_(Nel_y),
         nodes_(nodes),
         conn_(conn),
         nelTotal_(nelTotal),
         nnpTotal_(nnpTotal){};

    int getTotalNodes();
    int getTotalElements();
    Vector2View getNodesVector();
    Int4View getConnectivity();
    void computeFractionalElementArea();
    double getTotalArea();
    DoubleView getFractionalElementAreas();

};

Mesh initializeSheathMesh(int Nel_x,
                          int Nel_y,
                          std::string coord_file);

KOKKOS_INLINE_FUNCTION
bool P2LCheck(Vector2 xp, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 e1 = v2-v1;
    Vector2 e2 = v3-v2;
    Vector2 e3 = v4-v3;
    Vector2 e4 = v1-v4;
    Vector2 p1 = xp-v1;
    Vector2 p2 = xp-v2;
    Vector2 p3 = xp-v3;
    Vector2 p4 = xp-v4;

    if (e1.cross(p1) < 0.0)
        return false;
    if (e2.cross(p2) < 0.0)
        return false;
    if (e3.cross(p3) < 0.0)
        return false;
    if (e4.cross(p4) < 0.0)
        return false;

    return true;
}

} // namespace sheath

#include "GitrmSheathTestUtils.hpp"

#endif
