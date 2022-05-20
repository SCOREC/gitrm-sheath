#ifndef GitrmSheathTestUtils_hpp
#define GitrmSheathTestUtils_hpp

#include "GitrmSheath.hpp"

namespace sheath{

class Particles{
private:
    int numParticles_;
    Mesh meshObj_;
    Vector2View positions_;
    IntView elementIDs_;

public:
    int getTotalParticles();
    Mesh getMeshObj();
    Vector2View getParticlePostions();
    IntView getParticleElementIDs();

    Particles(){};

    Particles(int numParticles,
              Mesh meshObj,
              Vector2View positions,
              IntView elementIDs):
              numParticles_(numParticles),
              meshObj_(meshObj),
              positions_(positions),
              elementIDs_(elementIDs){};

};

Particles initializeParticles(int numParticles, Mesh meshObj, unsigned int rngSeed);

}

#endif
