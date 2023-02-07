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

#define maxVerti 8
#define maxParts 8
#define maxElemsPerVert 5

using Vector2View = Kokkos::View<Vector2*>;
using Int4View = Kokkos::View<int*[maxVerti+1]>;
using IntElemsPerVertView = Kokkos::View<int*[maxElemsPerVert+1]>;
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
    Int4View elemFaceBdry_;
    int nnpTotal_;
    int nelTotal_;

    double totArea_;
    DoubleView fracArea_;

    Vector2View Efield_;
    
    IntView elem2Particles_;
    IntElemsPerVertView vertex2Elems_;
public:
    Mesh(){};

    Mesh(int Nel_x,
         int Nel_y,
         Vector2View nodes,
         Int4View conn,
         Int4View elemFaceBdry,
         int nelTotal,
         int nnpTotal,
         Vector2View Efield,
         IntView elem2Particles = IntView("notInitElem2Particles",0),
         IntElemsPerVertView vertex2Elems = IntElemsPerVertView("notInitVertex2Elems",0)):
         Nel_x_(Nel_x),
         Nel_y_(Nel_y),
         nodes_(nodes),
         conn_(conn),
         elemFaceBdry_(elemFaceBdry),
         nelTotal_(nelTotal),
         nnpTotal_(nnpTotal),
         Efield_(Efield),
         elem2Particles_(elem2Particles),
         vertex2Elems_(vertex2Elems){};


    int getTotalNodes();
    int getTotalElements();
    int getTotalXElements();
    int getTotalYElements();
    Vector2View getNodesVector();
    Vector2View getEfieldVector();
    Int4View getConnectivity();
    Int4View getElemFaceBdry();
    void computeFractionalElementArea();
    double getTotalArea();
    DoubleView getFractionalElementAreas();
    
    IntView getElem2Particles();
    IntElemsPerVertView getVertex2Elems();
    void setElem2Particles(IntView elem2Particles);
    void setVertex2Elems(IntElemsPerVertView vertex2Elems);
};

Mesh initializeSheathMesh(int Nel_x,
                          int Nel_y,
                          std::string coord_file,
                          std::string Efield_file);

Mesh initializeSimpleMesh();

Mesh initializeTestMesh(int factor);

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
FaceDir T2LCheck(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    Vector2 xnew = xp+dx;
    Vector2 p1 = v1-xp;
    Vector2 p2 = v2-xp;
    Vector2 p3 = v3-xp;
    Vector2 p4 = v4-xp;
    Vector2 e1 = v2-v1;
    Vector2 e2 = v3-v2;
    Vector2 e3 = v4-v3;
    Vector2 e4 = v1-v4;

    double p1xdx = p1.cross(dx);
    double p2xdx = p2.cross(dx);
    if (p1xdx*p2xdx < 0.0){
        if (e1.cross(xnew-v1) < 0.0)
            return south;
    }
    double p3xdx = p3.cross(dx);
    if (p2xdx*p3xdx < 0.0){
        if (e2.cross(xnew-v2) < 0.0)
            return east;
    }
    double p4xdx = p4.cross(dx);
    if (p3xdx*p4xdx < 0.0){
        if (e3.cross(xnew-v3) < 0.0)
            return north;
    }
    if (p4xdx*p1xdx < 0.0){
        if (e4.cross(xnew-v4) < 0.0)
            return west;
    }
    return none;
}

KOKKOS_INLINE_FUNCTION
Vector2 findIntersectionPoint(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2){
    Vector2 fCenter = (v1+v2)*0.5;
    Vector2 fNormal = (v2-v1).rotateCW90();
    double lambda = ((fCenter-xp).dot(fNormal))/(dx.dot(fNormal));
    return (xp + dx*lambda);
}

KOKKOS_INLINE_FUNCTION
double findIntersectionLambda(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2){
    Vector2 fCenter = (v1+v2)*0.5;
    Vector2 fNormal = (v2-v1).rotateCW90();
    return ((fCenter-xp).dot(fNormal))/(dx.dot(fNormal));
}

KOKKOS_INLINE_FUNCTION
void getCoeffsForQuadBC(Vector2 xp, Vector2 v1,
                        Vector2 v2, Vector2 v3,
                        Vector2 v4, double* lambda, double* mu){
    Vector2 a = v1-xp;
    Vector2 b = v2-v1;
    Vector2 c = v4-v1;
    Vector2 d = v1-v2+v3-v4;
    double x=0.1;
    double y=0.1;
    double tol=1e-5;
    Vector2 f, Df1, Df2;
    double det, dx, dy, norm;
    norm = sqrt(x*x + y*y);
    int iter=0;
    int iter_max = 20;
    while (norm>tol && iter<iter_max){
        f = a + b*x + c*y + d*x*y;
        Df1 = b + d*y;
        Df2 = c + d*x;
        det = 1.0/(Df1[0]*Df2[1]-Df1[1]*Df2[0]);
        dx = det*(-Df2[1]*f[0] + Df2[0]*f[1]);
        dy = det*( Df1[1]*f[0] - Df1[0]*f[1]);
        x+=dx;
        y+=dy;
        norm = sqrt(dx*dx + dy*dy);
        iter++;
        if (norm>10.0)
            iter = iter_max;
    }

    if (iter < iter_max){
        *lambda = x;
        *mu = y;
        // printf("Newton-Raphson CONVERGED -- norm=%2.5e  iter=%d\n",norm,iter);
    }
    else{
        printf("Newton-Raphson DID NOT CONVERGE -- norm=%2.5e\n",norm);
    }
}

KOKKOS_INLINE_FUNCTION
void getWachpressCoeffs(Vector2 xp, Vector2 v1,
		        Vector2 v2, Vector2 v3,
		        Vector2 v4, double* l1, 
			double* l2, double* l3,
			double* l4){

   Vector2 e1 = v2-v1;
   Vector2 e2 = v3-v2;
   Vector2 e3 = v4-v3;
   Vector2 e4 = v1-v4;
   Vector2 p1 = xp-v1;
   Vector2 p2 = xp-v2;
   Vector2 p3 = xp-v3;
   Vector2 p4 = xp-v4;
   double p1mag = p1.magnitudesq();
   double p2mag = p2.magnitudesq();
   double p3mag = p3.magnitudesq();
   double p4mag = p4.magnitudesq();
   double wsum = 0.0;

   // cot of angle b/w v2-v1-xp
   double d = e1.dot(p1)/e1.cross(p1);
   // cot of angle b/w xp-v1-v4
   double g = p1.dot(e4)/p1.cross(e4);
   double w1 = (d+g)/p1mag;
   wsum += w1;
   
   // cot of angle b/w v3-v2-xp
   d = e2.dot(p2)/e2.cross(p2);
   // cot of angle b/w xp-v2-v1
   g = p2.dot(e1)/p2.cross(e1);
   double w2 = (d+g)/p2mag;
   wsum += w2;
   
   // cot of angle b/w v4-v3-xp
   d = e3.dot(p3)/e3.cross(p3);
   // cot of angle b/w xp-v3-v2
   g = p3.dot(e2)/p3.cross(e2);
   double w3 = (d+g)/p3mag;
   wsum += w3;
   
   // cot of angle b/w v1-v4-xp
   d = e4.dot(p4)/e4.cross(p4);
   // cot of angle b/w xp-v4-v3
   g = p4.dot(e3)/p4.cross(e3);
   double w4 = (d+g)/p4mag;
   wsum += w4;
   

   *l1 = w1/wsum;
   *l2 = w2/wsum;
   *l3 = w3/wsum;
   *l4 = w4/wsum;


}


KOKKOS_INLINE_FUNCTION
void getWachpressCoeffs(Vector2 xp, 
                        int numVerti,
			Vector2* v,
                        double* l){
    // n max is conn_size
    //   0 1 ... n      vn = nodes(conn(parti_iel,i))   :Vector2[]
    //   0 1 ... n 0    vn shifts 				[+1]
    //    0 1 ... n     edges                            :Vector2[]
    //  0 1 ... n       p = xp - nodes(conn(parti_iel,i)):Vector2[]
    //  0 1 ... n       weights                          :Double2[]
    //  ===================================================================
    //  rearrange the edges n 1 2 ... n-1
    //                   = double d = e(i).dot(p(i))/e(i).cross(p(i));
    //                     double g = p(i).dot(e(???))/p(i).cross(e(???));
    //                                          ??? = n 1 2 ... n-1
    //                     double w(i) = (d+g)/p(i).magnitudesq();
    //                     double wsum += w(i);
    Vector2 e[maxVerti+1];
    Vector2 p[maxVerti];
    double w[maxVerti];
    int i;
    for(i = 0; i<numVerti; i++){
        e[i+1] = v[i+1] - v[i];
        p[i] = xp - v[i];     
    } 
    e[0] = e[numVerti];
    double d,g, wsum = 0;
    for(i = 0; i<numVerti; i++){
        d = e[i+1].dot(p[i])/e[i+1].cross(p[i]);
        g = p[i].dot(e[i])/p[i].cross(e[i]);
        w[i] = (d+g)/p[i].magnitudesq();
        wsum += w[i];
    }
    for(i = 0; i<numVerti; i++){
        l[i] = w[i]/wsum;
    }
}

KOKKOS_INLINE_FUNCTION
void getWachpressCoeffsByArea(Vector2 xp,
                              int numVerti,
			      Vector2* v, 
                              double* l){
    Vector2 e[maxVerti+1];
    Vector2 p[maxVerti];
    double w[maxVerti];
    ///*
    for(int i = 0; i<numVerti; i++){
        e[i+1] = v[i+1] -v[i];
        //if(numVerti == 3){
        //    printf("e[%d](%.3f,%.3f)\n",i+1, e[i+1][0],e[i+1][1]);
        //}
        p[i] = v[i] - xp;
    }
    e[0] = e[numVerti];
    //if(numVerti == 3){
        //printf("e[%d](%.3f,%.3f)\n",0, e[0][0],e[0][1]);
    //}
    
    double c[maxVerti];
    double a[maxVerti];
    for(int i = 0; i<numVerti; i++){
        c[i] = e[i].cross(e[i+1]);
        a[i] = p[i].cross(e[i+1]);
       // printf("c: %.3f, a: %.3f\n",c[i],a[i]);
    }
    double wSum = 0.0;
    for(int i = 0; i<numVerti; i++){
        double aProduct = 1.0;
        for(int j = 0;j<numVerti-2; j++){
            //if(i == j or i-1 == j) aProduct /= a[j];
            aProduct *= a[(j+i+1)%numVerti];// each vertex goes to vertexNumber(i)-2
            //printf("numVerti: %d,i: %d, j: %d, i+j+1|numVerti: %d\n", numVerti, i, j,(i+j+1)%numVerti);
        }
        w[i] = c[i] * aProduct;
        wSum += w[i];
    }
    //*/
    for(int i = 0; i<numVerti; i++){
        l[i] = w[i]/wSum;
        //if(numVerti == 3){
        //    printf("w[%d],wSum = %.3f, %.3f \n",i, w[i], wSum);
        //}
    }
}

KOKKOS_INLINE_FUNCTION
void gradient(Vector2 xp, int numVerti, Vector2* v, double* l){
    Vector2 e[maxVerti+1];
    Vector2 p[maxVerti+1];
    for(int i = 0; i<numVerti; i++){
        e[i+1] = v[i+1] -v[i];
        p[i+1] = v[i+1] - xp;
    }
    e[0] = e[numVerti];
    p[0] = p[numVerti];
    
    double h[maxVerti+1];   
    Vector2 n[maxVerti+1];
    
    for(int i=0; i<numVerti+1; i++){
        n[i][0] = e[i][1];
        n[i][1] = -e[i][0];
        //timing ?
        h[i] = p[i].dot(n[i]);
        //check cross product of n.cross(e) positive
        //printf("%d: n.cross(e)= %.3f\n",i,n[i].cross(e[i]));
    }

    double w[maxVerti];
    double wSum = 0.0;
    for(int i=0; i<numVerti; i++){
        w[i] = n[i].cross(n[i+1])/(h[i]*h[i+1]);
        wSum += w[i];
        //printf("w[%d]= %.3f\n",i,w[i]);
    }   

    for(int i = 0; i<numVerti; i++){
        l[i] = w[i]/wSum;
        //if(numVerti == 3){
        //    printf("w[%d],wSum = %.3f, %.3f \n",i, w[i], wSum);
        //}
    }
    
    //finish the gradient calc
    Vector2 gradientW[maxVerti];
    Vector2 ratio[maxVerti];
    for(int i=0;i<numVerti;i++){
        gradientW[i] = Vector2(w[i]*(n[i][0]/h[i]+n[i+1][0]/h[i+1]),w[i]*(n[i][1]/h[i]+n[i+1][1]/h[i+1]));
        ratio[i] = Vector2(gradientW[i][0]/w[i],gradientW[i][1]);
        printf("gradientW[%d]= (%1.3e,%1.3e)\t| ratio[%d]= (%1.3e,%1.3e)\n",i,gradientW[i][0],gradientW[i][1],i,ratio[i][0],ratio[i][1]);
    }
}

KOKKOS_INLINE_FUNCTION
void getTriangleBC(Vector2 xp, Vector2 v1,
                    Vector2 v2, Vector2 v3,
                    Vector2 v4, double* lambda0,
                    double* lambda1, double* lambda2,
                    double* lambda3){
    Vector2 e1 = v2-v1;
    Vector2 e2 = v3-v2;
    Vector2 e3 = v4-v3;
    Vector2 e4 = v1-v4;
    Vector2 e5 = v4-v2;
    Vector2 p1 = xp-v1;
    Vector2 p2 = xp-v2;
    Vector2 p3 = xp-v3;
    Vector2 p4 = xp-v4;
    double triArea;
    if (e5.cross(xp-v2) < 0.0){
        // Upper-Tri
        triArea = e3.cross(e2);
        *lambda0 = 0.0;
        *lambda1 = fabs(e3.cross(p3)/triArea);
        *lambda2 = fabs(p2.cross(e5)/triArea);
        *lambda3 = 1.0-(*lambda2)-(*lambda1);
    }
    else{
        // Lower-Tri
        triArea = e1.cross(-e4);
        *lambda0 = fabs(e5.cross(p2)/triArea);
        *lambda1 = fabs(e4.cross(p4)/triArea);
        *lambda2 = 0.0;
        *lambda3 = 1.0-(*lambda0)-(*lambda1);
    }

}

KOKKOS_INLINE_FUNCTION
FaceDir minValueFaceDir(double l1, double l2, double l3, double l4){
    if (l1 <= l2){
        if (l1 <= l3){
            if (l1 <= l4){
                return south;
            }
            else{
                return west;
            }
        }
        else{
            if (l3 <= l4){
                return north;
            }
            else{
                return west;
            }
        }
    }
    else {
        if (l2 <= l3){
            if (l2 <= l4){
                return east;
            }
            else{
                return west;
            }
        }
        else{
            if (l3 <= l4){
                return north;
            }
            else {
                return west;
            }
        }
    }
    return none;
}

KOKKOS_INLINE_FUNCTION
FaceDir MacphersonCheckAllFace(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    bool crossedEast = false;
    bool crossedWest = false;
    bool crossedNorth = false;
    bool crossedSouth = false;
    double l1 = findIntersectionLambda(xp,dx,v1,v2);
    if (l1 > 0.0 && l1 < 1.0){
        crossedSouth = true;
    }
    else{
        l1 = 1.0;
    }
    double l2 = findIntersectionLambda(xp,dx,v2,v3);
    if (l2 > 0.0 && l2 < 1.0){
        crossedEast = true;
    }
    else{
        l2 = 1.0;
    }
    double l3 = findIntersectionLambda(xp,dx,v3,v4);
    if (l3 > 0.0 && l3 < 1.0){
        crossedNorth = true;
    }
    else{
        l3 = 1.0;
    }
    double l4 = findIntersectionLambda(xp,dx,v4,v1);
    if (l4 > 0.0 && l4 < 1.0){
        crossedWest = true;
    }
    else{
        l4 = 1.0;
    }

    if (!crossedEast && !crossedWest && !crossedNorth && !crossedSouth){
        return none;
    }
    else{
        return minValueFaceDir(l1,l2,l3,l4);
    }
}

KOKKOS_INLINE_FUNCTION
FaceDir MacphersonCheckForEastEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    bool crossedEast = false;
    bool crossedNorth = false;
    bool crossedSouth = false;
    double l1 = findIntersectionLambda(xp,dx,v1,v2);
    if (l1 > 0.0 && l1 < 1.0){
        crossedSouth = true;
    }
    else{
        l1 = 1.0;
    }
    double l2 = findIntersectionLambda(xp,dx,v2,v3);
    if (l2 > 0.0 && l2 < 1.0){
        crossedEast = true;
    }
    else{
        l2 = 1.0;
    }
    double l3 = findIntersectionLambda(xp,dx,v3,v4);
    if (l3 > 0.0 && l3 < 1.0){
        crossedNorth = true;
    }
    else{
        l3 = 1.0;
    }
    double l4 = 1.0;

    if (!crossedEast && !crossedNorth && !crossedSouth){
        return none;
    }
    else{
        return minValueFaceDir(l1,l2,l3,l4);
    }
}

KOKKOS_INLINE_FUNCTION
FaceDir MacphersonCheckForWestEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    bool crossedWest = false;
    bool crossedNorth = false;
    bool crossedSouth = false;
    double l1 = findIntersectionLambda(xp,dx,v1,v2);
    if (l1 > 0.0 && l1 < 1.0){
        crossedSouth = true;
    }
    else{
        l1 = 1.0;
    }
    double l2 = 1.0;

    double l3 = findIntersectionLambda(xp,dx,v3,v4);
    if (l3 > 0.0 && l3 < 1.0){
        crossedNorth = true;
    }
    else{
        l3 = 1.0;
    }
    double l4 = findIntersectionLambda(xp,dx,v4,v1);
    if (l4 > 0.0 && l4 < 1.0){
        crossedWest = true;
    }
    else{
        l4 = 1.0;
    }

    if (!crossedWest && !crossedNorth && !crossedSouth){
        return none;
    }
    else{
        return minValueFaceDir(l1,l2,l3,l4);
    }
}

KOKKOS_INLINE_FUNCTION
FaceDir MacphersonCheckForSouthEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    bool crossedEast = false;
    bool crossedWest = false;
    bool crossedSouth = false;
    double l1 = findIntersectionLambda(xp,dx,v1,v2);
    if (l1 > 0.0 && l1 < 1.0){
        crossedSouth = true;
    }
    else{
        l1 = 1.0;
    }
    double l2 = findIntersectionLambda(xp,dx,v2,v3);
    if (l2 > 0.0 && l2 < 1.0){
        crossedEast = true;
    }
    else{
        l2 = 1.0;
    }
    double l3 = 1.0;
    double l4 = findIntersectionLambda(xp,dx,v4,v1);
    if (l4 > 0.0 && l4 < 1.0){
        crossedWest = true;
    }
    else{
        l4 = 1.0;
    }

    if (!crossedEast && !crossedWest && !crossedSouth){
        return none;
    }
    else{
        return minValueFaceDir(l1,l2,l3,l4);
    }
}

KOKKOS_INLINE_FUNCTION
FaceDir MacphersonCheckForNorthEntry(Vector2 xp, Vector2 dx, Vector2 v1, Vector2 v2, Vector2 v3, Vector2 v4){
    bool crossedEast = false;
    bool crossedWest = false;
    bool crossedNorth = false;
    double l1 = 1.0;
    double l2 = findIntersectionLambda(xp,dx,v2,v3);
    if (l2 > 0.0 && l2 < 1.0){
        crossedEast = true;
    }
    else{
        l2 = 1.0;
    }
    double l3 = findIntersectionLambda(xp,dx,v3,v4);
    if (l3 > 0.0 && l3 < 1.0){
        crossedNorth = true;
    }
    else{
        l3 = 1.0;
    }
    double l4 = findIntersectionLambda(xp,dx,v4,v1);
    if (l4 > 0.0 && l4 < 1.0){
        crossedWest = true;
    }
    else{
        l4 = 1.0;
    }

    if (!crossedEast && !crossedWest && !crossedNorth){
        return none;
    }
    else{
        return minValueFaceDir(l1,l2,l3,l4);
    }
}


KOKKOS_INLINE_FUNCTION
void initArrayWith(Vector2 arr[],int n, Vector2 fill){
    for(int i=0; i<n; i++){
        arr[i] = fill;
    }
}

KOKKOS_INLINE_FUNCTION
void initArrayWith(double arr[],int n, double fill){
    for(int i=0; i<n; i++){
        arr[i] = fill;
    }
}

} // namespace sheath

#include "GitrmSheathTestUtils.hpp"

#endif
