/*
 * meanShift extracts modes at specific locations.
 * Compile it in MATLAB by:
 *      mex meanShift.cpp
 * Function call in MATLAB:
 *      finals = meanShift(points, initializations, sigma, radius, tolerance, maxiters)
 * where
 * Input:
 *       - points: point cloud in d-dimension of size n x d
 *       - initializations: starting locations in d-dimension of size m x d
 *       - sigma: kernel parameter
 *       - radius: window parameter
 *       - tolerance: numerical tolerance for convergence checking
 *       - maxiters: the maximum number of iterations
 * Output:
 *       - finals: convergent locations in d-dimension of size m x d
*/

#include "mex.h"
#include <math.h>
#include <vector>

inline int sub2ind(int & nRows, int & nCols, int i, int j)
{
    return j * nRows + i;
}

inline void ind2sub(int & nRows, int & nCols, int k, int & i, int & j)
{
    i = k % nRows;
    j = k / nRows;
}

double squaredDifference(int & nPoints, int & nDims, double *& points, int & i, int & nQuerries, double *& initializations, int & j)
{
    double result = 0;
    for (int k = 0; k < nDims; ++k)
        result += pow(points[sub2ind(nPoints, nDims, i, k)] - initializations[sub2ind(nQuerries, nDims, j, k)], 2);
    return result;
}

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    int nPoints = (int) mxGetM(prhs[0]);
    int nDims = (int) mxGetN(prhs[0]);
    double * points = mxGetPr(prhs[0]);
    int nQuerries = (int) mxGetM(prhs[1]);
    double * initializations = new double[nQuerries * nDims * sizeof(double)];
    memcpy(initializations, mxGetPr(prhs[1]), nQuerries * nDims * sizeof(double));
    double sigma2 = mxGetScalar(prhs[2]) * mxGetScalar(prhs[2]);
    double radius2 = mxGetScalar(prhs[3]) * mxGetScalar(prhs[3]);
    double tolerance = mxGetScalar(prhs[4]);
    int maxiters = (int) mxGetScalar(prhs[5]);
    plhs[0] = mxCreateDoubleMatrix(nQuerries, nDims, mxREAL);
    double * finals = mxGetPr(plhs[0]);
    memcpy(finals, initializations, nQuerries * nDims * sizeof(double));
    double * distances = new double[nPoints];
    for (int loop = 0; loop < nQuerries; ++loop)
    {
        int iters = 0;
        while (iters < maxiters)
        {
            bool flag = false;
            double denominator = 0;
            for (int i = 0; i < nPoints; ++i)
            {
                distances[i] = squaredDifference(nPoints, nDims, points, i, nQuerries, initializations, loop);
                if (distances[i] <= radius2)
                {
                    flag = true;
                    denominator += exp(-distances[i] / sigma2);
                }
            }
            if (!flag)
                break;
            for (int j = 0; j < nDims; ++j)
                finals[sub2ind(nQuerries, nDims, loop, j)] = 0;
            for (int i = 0; i < nPoints; ++i)
                if (distances[i] <= radius2)
                {
                    for (int j = 0; j < nDims; ++j)
                        finals[sub2ind(nQuerries, nDims, loop, j)] += exp(-distances[i] / sigma2) * points[sub2ind(nPoints, nDims, i, j)];
                }
            for (int j = 0; j < nDims; ++j)
                finals[sub2ind(nQuerries, nDims, loop, j)] /= denominator;
            if (sqrt(squaredDifference(nQuerries, nDims, finals, loop, nQuerries, initializations, loop)) < tolerance)
                break;
            iters = iters + 1;
            for (int j = 0; j < nDims; ++j)
                initializations[sub2ind(nQuerries, nDims, loop, j)] = finals[sub2ind(nQuerries, nDims, loop, j)];
        }
    }
    delete [] distances;
    delete [] initializations;
}
