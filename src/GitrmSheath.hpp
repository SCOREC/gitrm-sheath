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

class Mesh{
private:
    int Nel_x_;
    int Nel_y_;
    Vector2View nodes_;
    Kokkos::View<int*[4]> conn_;

public:
    Mesh(){};

    Mesh(int Nel_x,
         int Nel_y,
         Vector2View nodes,
         Kokkos::View<int*[4]> conn):
         Nel_x_(Nel_x),
         Nel_y_(Nel_y),
         nodes_(nodes),
         conn_(conn){};
};

Mesh dummy();


} // namespace sheath

#endif
