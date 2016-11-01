/*
 * This mex file is to estimate the normal vectors for a 2D point cloud based on PCA.
 * To compile it from MATLAB, use the following command:
 * 	Windows:
 *      mex('-v', '-largeArrayDims', 'estimateNormals.cpp', fullfile(matlabroot, 'extern', 'lib', computer('arch'), 'microsoft', 'libmwlapack.lib'));
 * 	Unix:
 * 		mex -v -largeArrayDims estimateNormals.cpp -lmwlapack
 * Call this function in MATLAB by
 *      normals = estimateNormals(points, neighbors);
 * where
 *      points: is the pointcloud with N 2-D points, an array N x 2
 *      neighbors: contains the neighborhood information, of size N x K where K is the number of neighbors for each data point (e.g. K = 5).
 *          One may obtain this information from MATLAB built-in knnsearch function, e.g. neighbors = cell2mat(knnsearch(points, points, 'K', K, 'IncludeTies', true));
 *      normals: is the normals associating with each point, is of size N x 2, all normals are unit vectors and point to the origin
*/

#if !defined(_WIN32)
#define dgesvd dgesvd_
#endif

#include "mex.h"
#include "lapack.h"
#include <string.h>
#include <stddef.h>

void princomp(double *& points, double *& neighbors, double *& eigenValues, double *& eigenVectors, double **& X, double U[], double VT[], double *& work, ptrdiff_t & lwork, int & q, int & N, int & K)
{
    double meanX = 0, meanY = 0;
    for (int i = 0; i < K; ++i)
    {
        X[i][0] = points[(int) neighbors[N * i + q] - 1];
        X[i][1] = points[N + (int) neighbors[N * i + q] - 1];
        meanX += X[i][0]; meanY += X[i][1];
    }
    X[K][0] = points[q]; X[K][1] = points[N + q];
    meanX += X[K][0]; meanY += X[K][1];
    meanX /= (double) (K + 1); meanY /= (double) (K + 1);
    for (int i = 0; i <= K; ++i)
    {
        X[i][0] -= meanX;
        X[i][1] -= meanY;
    }
    memset(eigenVectors, 0, 4 * sizeof(double));
    for (int i = 0; i <= K; ++i)
    {
        eigenVectors[0] += X[i][0] * X[i][0];
        eigenVectors[1] += X[i][0] * X[i][1];
        eigenVectors[3] += X[i][1] * X[i][1];
    }
    eigenVectors[0] /= (double) K; eigenVectors[1] /= (double) K; eigenVectors[2] /= (double) K; eigenVectors[3] /= (double) K;
    eigenVectors[2] = eigenVectors[1];
    char jobu = 'O', jobvt = 'N';
    ptrdiff_t m = 2, n = 2, lda = 2, ldu = 2, ldvt = 2, info;
    dgesvd(&jobu, &jobvt, &m, &n, eigenVectors, &lda, eigenValues, U, &ldu, VT, &ldvt, work, &lwork, &info);
}

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    int N = (int) mxGetM(prhs[1]);
    int K = (int) mxGetN(prhs[1]);
    double * inPoints = mxGetPr(prhs[0]);
    double * inNeighbors = mxGetPr(prhs[1]);
    plhs[0] = mxCreateDoubleMatrix(N, 2, mxREAL);
    double * outNormals = mxGetPr(plhs[0]);
    double * eigenValues = new double[2];
    double * eigenVectors = new double[4];
    ptrdiff_t lwork = 100;
    double * work = new double[lwork];
    double ** X = new double *[K + 1];
    for (int i = 0; i <= K; ++i)
        X[i] = new double[2];
    double * U = new double[4];
    double * VT = new double[4];
    for (int i = 0; i < N; ++i)
    {
        princomp(inPoints, inNeighbors, eigenValues, eigenVectors, X, U, VT, work, lwork, i, N, K);
        outNormals[i] = eigenVectors[2];
        outNormals[N + i] = eigenVectors[3];
        if (outNormals[i] * inPoints[i] + outNormals[N + i] * inPoints[N + i] > 0)
        {
            outNormals[i] = -outNormals[i];
            outNormals[N + i] = -outNormals[N + i];
        }
    }
    for (int i = 0; i <= K; ++i)
        delete [] X[i];
    delete [] U;
    delete [] VT;
    delete [] X;
    delete [] work;
    delete [] eigenValues;
    delete [] eigenVectors;
}
