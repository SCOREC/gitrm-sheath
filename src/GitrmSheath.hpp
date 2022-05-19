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

class Mesh{
public:
    // Kokkos::View<Vector2*> nodes;
    // Kokkos::View<int*[4]> conn;

    Mesh(){};

};

Mesh dummy();


} // namespace sheath

#endif
