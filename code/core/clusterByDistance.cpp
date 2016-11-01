/*
 * clusterByDistance clusters data into groups according to the follwing procedure. Starting with an unlabeled data point x, all data points within "distance" threshold are labeled, called expansion.
 * The cluster center is formed. This cluster keeps being expanded until no more member. The process starts over with other data points.
 * Compile it in MATLAB by:
 *      mex clusterByDistance.cpp
 * Function call in MATLAB:
 *      centers = clusterByDistance(points, neighbors, threshold)
 * where
 * Input:
 *       - points: point cloud in d-dimension of size n x d
 *       - neighbors: neighborhood information, of size n x k
 *       - threshold: distance to assign membership
 * Output:
 *       - centers: final cluster means, of size m x d
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

void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
    int nPoints = (int) mxGetM(prhs[0]);
    int nDims = (int) mxGetN(prhs[0]);
    double * points = mxGetPr(prhs[0]);
    int K = (int) mxGetN(prhs[1]);
    double * neighbors = mxGetPr(prhs[1]);
    double threshold = mxGetScalar(prhs[2]);
    std::vector<double*> centers;
    std::vector<int> counts;
    centers.clear();
    counts.clear();
    bool * labeled = new bool[nPoints];
    memset(labeled, 0, nPoints * sizeof(bool));
    int * queue = new int[nPoints];
    for (int i = 0; i < nPoints; ++i)
    {
        if (!labeled[i])
        {
            int front = 0, rear = 0;
            queue[front] = i;
            labeled[i] = true;
            centers.push_back(new double[nDims]);
            counts.push_back(1);
            for (int k = 0; k < nDims; ++k)
                centers[centers.size() - 1][k] = points[sub2ind(nPoints, nDims, i, k)];
            while (front <= rear)
            {
                int u = queue[front++];
                for (int j = 0; j < K; ++j)
                {
                    int v = neighbors[sub2ind(nPoints, K, u, j)] - 1;
                    if (!labeled[v])
                    {
                        double d = 0;
                        for (int k = 0; k < nDims; ++k)
                            d += pow(centers[centers.size() - 1][k] / counts[centers.size() - 1] - points[sub2ind(nPoints, nDims, v, k)], 2);
                        d = sqrt(d);
                        if (d <= threshold)
                        {
                            ++counts[centers.size() - 1];
                            for (int k = 0; k < nDims; ++k)
                                centers[centers.size() - 1][k] += points[sub2ind(nPoints, nDims, v, k)];
                            queue[++rear] = v;
                            labeled[v] = true;
                        }
                    }
                }
            }
        }
    }
    delete [] queue;
    delete [] labeled;
    centers.shrink_to_fit();
    counts.shrink_to_fit();
    int m = (int) centers.size();
    plhs[0] = mxCreateDoubleMatrix(m, nDims, mxREAL);
    double * out = mxGetPr(plhs[0]);
    for (int i = 0; i < centers.size(); ++i)
    {
        for (int j = 0; j < nDims; ++j)
            out[sub2ind(m, nDims, i, j)] = centers[i][j] / counts[i];
        delete [] centers[i];
    }
    centers.resize(0);
    counts.resize(0);
}
