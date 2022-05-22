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
using BoolView = Kokkos::View<bool*>;

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
    int getTotalXElements();
    int getTotalYElements();
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

KOKKOS_INLINE_FUNCTION
FaceDir T2LCheckAllFace(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    if (p1xdx*p2xdx < 0.0)
        return south;
    double p3xdx = p3.cross(dx);
    if (p2xdx*p3xdx < 0.0)
        return east;
    double p4xdx = p4.cross(dx);
    if (p3xdx*p4xdx < 0.0)
        return north;
    if (p4xdx*p1xdx < 0.0)
        return west;

    return none;
}

KOKKOS_INLINE_FUNCTION
FaceDir T2LCheckForSouthEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    double p3xdx = p3.cross(dx);
    if (p2xdx*p3xdx < 0.0)
        return east;
    double p4xdx = p4.cross(dx);
    if (p3xdx*p4xdx < 0.0)
        return north;
    if (p4xdx*p1xdx < 0.0)
        return west;

    return none;
}

KOKKOS_INLINE_FUNCTION
FaceDir T2LCheckForWestEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    if (p1xdx*p2xdx < 0.0)
        return south;
    double p3xdx = p3.cross(dx);
    if (p2xdx*p3xdx < 0.0)
        return east;
    double p4xdx = p4.cross(dx);
    if (p3xdx*p4xdx < 0.0)
        return north;

    return none;
}

KOKKOS_INLINE_FUNCTION
FaceDir T2LCheckForNorthEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    if (p1xdx*p2xdx < 0.0)
        return south;
    double p3xdx = p3.cross(dx);
    if (p2xdx*p3xdx < 0.0)
        return east;
    double p4xdx = p4.cross(dx);
    if (p4xdx*p1xdx < 0.0)
        return west;

    return none;
}

KOKKOS_INLINE_FUNCTION
FaceDir T2LCheckForEastEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    if (p1xdx*p2xdx < 0.0)
        return south;
    double p3xdx = p3.cross(dx);
    double p4xdx = p4.cross(dx);
    if (p3xdx*p4xdx < 0.0)
        return north;
    if (p4xdx*p1xdx < 0.0)
        return west;

    return none;
}

KOKKOS_INLINE_FUNCTION
Vector2 findIntersectionPoint(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2){
    Vector2 fCenter = (v1+v2)*0.5;
    Vector2 fNormal = (v2-v1).rotateCW90();
    double lambda = ((fCenter-xp).dot(fNormal))/(dx.dot(fNormal));
    return (xp + dx*lambda);
}

} // namespace sheath

#include "GitrmSheathTestUtils.hpp"

#endif
