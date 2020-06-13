// DBSCAN base code provided by https://github.com/james-yoo/DBSCAN

#ifndef DBSCAN_H
#define DBSCAN_H

#include <vector>
#include <cmath>
#include "constants.h"

#define UNCLASSIFIED -1
#define CORE_POINT 1
#define BORDER_POINT 2
#define NOISE -2
#define SUCCESS 0
#define FAILURE -3

using namespace std;

typedef struct ClusterBBox
{
    int left, right, top, bottom;
    int clusterID;  // clustered ID
};

class DBSCAN {
public:    
    DBSCAN(int minPts, float eps, vector<ClusterBBox> boxes){
        m_minPoints = minPts;
        m_epsilon = eps;
        m_boxes = boxes;
    }
    ~DBSCAN(){}

    int run();
    vector<ClusterBBox> getBoxes();
    vector<int> calculateCluster(ClusterBBox point);
    int expandCluster(ClusterBBox point, int clusterID);
    inline double calculateDistance(ClusterBBox pointCore, ClusterBBox pointTarget);

    int getMinimumClusterSize() {return m_minPoints;}
    int getEpsilonSize() {return m_epsilon;}

private:
    vector<ClusterBBox> m_boxes;
    int m_minPoints;
    float m_epsilon;
};

#endif // DBSCAN_H
