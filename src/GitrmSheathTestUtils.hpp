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
    BoolView status_;

public:
    int getTotalParticles();
    Mesh getMeshObj();
    Vector2View getParticlePostions();
    IntView getParticleElementIDs();
    BoolView getParticleStatus();
    int computeTotalActiveParticles();
    void validateP2LAlgo();
    void validateP2LAlgoAlt();
    void T2LTracking(Vector2View dx);
    void T2LTrackingDebug(Vector2View dx);
    void MacphersonTracking(Vector2View dx);
    void interpolateQuadEField();
    void interpolateTriEField();
    void interpolateWachpress();
    Particles(){};

    Particles(int numParticles,
              Mesh meshObj,
              Vector2View positions,
              IntView elementIDs,
              BoolView status):
              numParticles_(numParticles),
              meshObj_(meshObj),
              positions_(positions),
              elementIDs_(elementIDs),
              status_(status){};

};

Particles initializeParticles(int numParticles, Mesh meshObj, unsigned int rngSeed);
Particles initializeSingleParticle(Mesh meshObj, unsigned int rngSeed);
Vector2View getRandDisplacements(int numParticles, int rngSeed, double scaleFactor);

}

#endif
