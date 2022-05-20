#ifndef GitrmSheathTestUtils_hpp
#define GitrmSheathTestUtils_hpp

#include "GitrmSheath.hpp"

namespace sheath{

class Particles{
private:
    int numParticles_;
    Mesh meshObj_;
    Vector2View positions_;

public:
    int getTotalParticles();
    Mesh getMeshObj();
    Vector2View getParticlePostions();

    Particles(){};

    Particles(int numParticles,
              Mesh meshObj,
              Vector2View positions:
              numParticles_(numParticles),
              meshObj_(meshObj),
              positions_(positions){};

    )

};

Particles initializeParticles(int numParticles, Mesh meshObj);

#endif
