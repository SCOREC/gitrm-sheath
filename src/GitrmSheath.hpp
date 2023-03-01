/*!
* \author Vignesh Vittal-Srinivasaragavan
* \date 05-19-2022
* \mainpage Sheath Electric Field Model for GITRm (with Kokkos)
*/

#ifndef GitrmSheath_hpp
#define GitrmSheath_hpp

#include <stdlib.h>
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

#include <netcdf.h>
#define ERRexit(e) {printf("Error: %s\n", nc_strerror(e)); exit(2);}

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

Mesh readMPASMesh(int ncid);

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
                              double* phi,
                              Vector2* gradientPhi){
    Vector2 e[maxVerti+1];
    Vector2 p[maxVerti];
    double w[maxVerti];
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
    
    double wdx[maxVerti];
    double wdy[maxVerti];
    initArrayWith(wdx,maxVerti,0.0);
    initArrayWith(wdy,maxVerti,0.0);
    double wdxSum = 0.0;
    double wdySum = 0.0;
    for(int i = 0; i<numVerti; i++){
        double aProduct = 1.0;
        for(int j = 0;j<numVerti-2; j++){
            int index1 = (j+i+1)%numVerti;
            aProduct *= a[index1];
            
            double productX = 1.0;
            double productY = 1.0;
            for(int k = 0; k<j; k++){
                int index2 = (i+k+1)%numVerti;
                productX *= a[index2];
                productY *= a[index2];
            }    
            productX *= -(v[index1+1][1]-v[index1][1]);
            productY *= v[index1+1][0]-v[index1][0];
            for(int k = j+1; k<numVerti-2; k++){
                int index2 = (i+k+1)%numVerti;
                productX *= a[index2];
                productY *= a[index2];
            }   
            wdx[i] += productX;
            wdy[i] += productY;
        }
        wdx[i] *= c[i];
        wdy[i] *= c[i];
        wdxSum += wdx[i];
        wdySum += wdy[i];
        w[i] = c[i] * aProduct;
        wSum += w[i];
    }
    
    double wSumInv = 1.0/wSum;
    for(int i = 0; i<numVerti; i++){
        phi[i] = w[i]*wSumInv;
        gradientPhi[i] = Vector2(wdx[i]*wSumInv-w[i]*wSumInv*wSumInv*wdxSum, wdy[i]*wSumInv-w[i]*wSumInv*wSumInv*wdySum);
        //if(numVerti == 3){
        //    printf("w[%d],wSum = %.3f, %.3f \n",i, w[i], wSum);
        //}
    }
}

KOKKOS_INLINE_FUNCTION
void gradientByHeight(Vector2 xp, int numVerti, Vector2* v, double* phi, Vector2* gradientPhi){
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
        phi[i] = w[i]/wSum;
        //if(numVerti == 3){
        //    printf("w[%d],wSum = %.3f, %.3f \n",i, w[i], wSum);
        //}
    }
    
    //finish the gradient calc
    //
    //TODO: gradient at vertex location
    //      new formular
    Vector2 ratio[maxVerti];
    Vector2 sumR;
    for(int i=0; i<numVerti; i++){
        ratio[i] =  Vector2((n[i][0]/h[i]+n[i+1][0]/h[i+1]),(n[i][1]/h[i]+n[i+1][1]/h[i+1]));
        sumR = Vector2((sumR[0]+phi[i]*ratio[i][0]),(sumR[1]+phi[i]*ratio[i][1]));
        //if(numVerti == 3)
        //    printf("%d: phi= %1.3f |sumR= (%1.3f,%1.3f)\t| ratio= (%1.3f,%1.3f)\n",i,phi[i],sumR[0],sumR[1],ratio[i][0],ratio[i][1]);
    }
    //printf("%d: phi= %1.3f |sumR= (%1.3f,%1.3f)\t| ratio= (%1.3f,%1.3f)\n",numVerti,phi[numVerti-1],sumR[0],sumR[1],ratio[numVerti-1][0],ratio[numVerti-1][1]);
    //Vector2 gradientPhi[maxVerti];
     
    for(int i=0; i<numVerti; i++){
        gradientPhi[i] = Vector2(phi[i]*(ratio[i][0]-sumR[0]), phi[i]*(ratio[i][1]-sumR[1]));
        //printf("%d: sumR= (%1.3f,%1.3f) |phi=%1.3f |ratio= (%7.3f,%7.3f) |gradientPhi= (%1.3f,%1.3f)\n",i,sumR[0],sumR[1],phi[i],ratio[i][0],ratio[i][1],gradientPhi[i][0],gradientPhi[i][1]);
        //printf("%d: gradientPhi= (%1.3f,%1.3f)\n",i,gradientPhi[i][0],gradientPhi[i][1]);
    }
}


KOKKOS_INLINE_FUNCTION
void gradientIntrepid(Vector2 xp, int numVerti, Vector2* v, double* phi, Vector2* gradientPhi){
    
    double numerator[maxVerti];
    double denominator = 0.0;
    Vector2 derivative[maxVerti];
    Vector2 derivative_sum;

    for(int k=0; k<numVerti; k++){
        int i = numVerti;
        if (k>0)
            i = k-1;
        int j = k+1;
        double a_k = 0.5*(v[i][0]*(v[k][1]-v[j][1]) - v[k][0]*(v[i][1]-v[j][1]) + v[j][0]*(v[i][1]-v[k][1]));
        
        double product = a_k;
        for(int m=0; m<numVerti; m++){
            i = m;
            j = m+1;
            if(m == numVerti-1)
                j=0;
            if(i!=k && j!=k){
                double a_ij = 0.5*(xp[0]*(v[j][1]-v[i][1]) - v[j][0]*(xp[1]-v[i][1]) + v[i][0]*(xp[1]-v[j][1]));
                product *= a_ij;
            }
        }
        numerator[k] = product;
        denominator += numerator[k];

        double product_sum[2] = {0.0,0.0};
        for(int m=0; m<numVerti; m++){
            i = m;
            j = m+1;
            if(m == numVerti-1)
                j = 0;
            
            if(i!=k && j!=k){
                double product_dx = a_k,product_dy = a_k;
                for(int l=0; l<numVerti; l++){
                    int s = l;
                    int t = l+1;
                    if(l == numVerti-1)
                        t=0;
                    if(s!=k && t!=k){
                        if(l!=m){
                            double a_st = 0.5*(xp[0]*(v[t][1]-v[s][1]) - v[t][0]*(xp[1]-v[s][1]) + v[s][0]*(xp[1]-v[t][1]));
                            product_dx *= a_st;
                            product_dy *= a_st;
                        }else{
                            double a_st_dx = 0.5*(v[t][1]-v[s][1]);
                            double a_st_dy = 0.5*(v[s][0]-v[t][0]);
                            product_dx *= a_st_dx;
                            product_dy *= a_st_dy;
                        }
                    }    
                }
                product_sum[0] += product_dx;
                product_sum[1] += product_dy;
            }
        }
        derivative[k] = Vector2(product_sum) ;
        derivative_sum = Vector2(derivative_sum[0] + derivative[k][0],derivative_sum[1] + derivative[k][1]);
    }
    
    for(int k=0; k<numVerti; k++){
        phi[k] = numerator[k]/denominator;
        gradientPhi[k] = Vector2(derivative[k][0]/denominator - (numerator[k]/(denominator*denominator))*derivative_sum[0],derivative[k][1]/denominator - (numerator[k]/(denominator*denominator))*derivative_sum[1]);       
    }   
}

KOKKOS_INLINE_FUNCTION
void gradientMPAS(Vector2 xp, int numVerti, Vector2* v, double* phi, Vector2* gradientPhi){
    // v change to 1 2 ... n 1 loop style
    double A[maxVerti+1];
    initArrayWith(A,maxVerti+1, 0.0);
    double B[maxVerti+1];
    initArrayWith(B,maxVerti+1, 0.0);
    double kappa[maxVerti];
    kappa[0] = 1.0;
    double l[maxVerti];
    
    for(int iVertex=0; iVertex<numVerti; iVertex++){
        int i1 = iVertex;
        int i2 = iVertex+1;
       
        // 0 = 1 -A[i]*x -B[i]*y (22)
        // cant solve the line through the origion 
        A[i2] = (v[i2][1]-v[i1][1])/(v[i1][0]*v[i2][1]-v[i2][0]*v[i1][1]); 
        B[i2] = (v[i1][0]-v[i2][0])/(v[i1][0]*v[i2][1]-v[i2][0]*v[i1][1]); 
        //A[i1] = (v[i2][1]-v[i1][1])/(v[i1][0]*v[i2][1]-v[i2][0]*v[i1][1]); 
        //B[i1] = (v[i1][0]-v[i2][0])/(v[i1][0]*v[i2][1]-v[i2][0]*v[i1][1]); 
   
    }
    A[0] = A[numVerti];
    B[0] = B[numVerti];
    //A[numVerti] = A[0];
    //B[numVerti] = B[0];
    
    for(int i=0; i<numVerti; i++)
        l[i] = 1 -A[i]*xp[0] -B[i]*xp[1];

    for(int iVertex=1; iVertex<numVerti;iVertex++){
        int im1 = iVertex-1;
        int i = iVertex;
        int ip1 = iVertex+1;
        // kappa equation (21)
        // the index is due to the shift in v[] which affect thh A[] and B[] index
        kappa[i] = kappa[im1]*(A[ip1]*(v[im1][0]-v[i][0])+B[ip1]*(v[im1][1]-v[i][1]))/ (A[im1]*(v[i][0]-v[im1][0])+B[im1]*(v[i][1]-v[im1][1]));
    }
    /*===print check
    if(numVerti == 7){
        for(int iVertex=0; iVertex<numVerti; iVertex++){
            printf("%d:A= %6.3f,B= %6.3f,kappa= %f \n",iVertex,A[iVertex],B[iVertex], kappa[iVertex]);
        }
    }//===*/

    double n[maxVerti];
    initArrayWith(n,maxVerti,1.0);
    double nSum = 0.0;

    double ndx[maxVerti];
    double ndy[maxVerti];
    initArrayWith(ndx,maxVerti,0.0);
    initArrayWith(ndy,maxVerti,0.0);
    double ndxSum = 0.0;
    double ndySum = 0.0;
    for(int i = 0; i<numVerti; i++){
        for(int j = 0;j<numVerti-2; j++){
            int index1 = (i+j+2)%numVerti;
            n[i] *= l[index1];
            double productX = 1.0;
            double productY = 1.0;
            for(int k = 0; k<j; k++){
                int index2 = (i+k+2)%numVerti;
                productX *= l[index2];     
                productY *= l[index2];   
            }
            productX *= -A[index1];
            productY *= -B[index1];
            for(int k = j+1; k<numVerti-2; k++){
                int index2 = (i+k+2)%numVerti;
                productX *= l[index2];     
                productY *= l[index2];
            }
            ndx[i] += productX;
            ndy[i] += productY;
        }
        n[i] *= kappa[i];
        nSum += n[i];
        ndx[i] *= kappa[i];
        ndy[i] *= kappa[i];
        ndxSum += ndx[i];
        ndySum += ndy[i];
    }


    double nSumInv = 1.0/nSum;
    for(int i=0; i<numVerti; i++){
        phi[i] = n[i]*nSumInv;
        gradientPhi[i] = Vector2(ndx[i]*nSumInv-n[i]*nSumInv*nSumInv*ndxSum,ndy[i]*nSumInv-n[i]*nSumInv*nSumInv*ndySum);
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



} // namespace sheath

#include "GitrmSheathTestUtils.hpp"

#endif
