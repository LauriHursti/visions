/* 
Base for DBSCAN provided by https://github.com/james-yoo/DBSCAN

MIT License

Copyright (c) 2018 james.yookh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "dbscan.h"
#include <math.h>

using std::min;
using std::max;

int DBSCAN::run()
{
    int clusterID = 1;
    vector<ClusterBBox>::iterator iter;
    for(iter = m_boxes.begin(); iter != m_boxes.end(); ++iter)
    {
        if ( iter->clusterID == UNCLASSIFIED )
        {
            if ( expandCluster(*iter, clusterID) != FAILURE )
            {
                clusterID += 1;
            }
        }
    }

    return 0;
}

int DBSCAN::expandCluster(ClusterBBox point, int clusterID)
{    
    vector<int> clusterSeeds = calculateCluster(point);

    if ( clusterSeeds.size() < m_minPoints )
    {
        point.clusterID = NOISE;
        return FAILURE;
    }
    else
    {
        int index = 0, indexCorePoint = 0;
        vector<int>::iterator iterSeeds;
        for( iterSeeds = clusterSeeds.begin(); iterSeeds != clusterSeeds.end(); ++iterSeeds)
        {
            m_boxes.at(*iterSeeds).clusterID = clusterID;
            if (m_boxes.at(*iterSeeds).left == point.left && m_boxes.at(*iterSeeds).right == point.right && m_boxes.at(*iterSeeds).top == point.top && m_boxes.at(*iterSeeds).bottom == point.bottom)
            {
                indexCorePoint = index;
            }
            ++index;
        }
        clusterSeeds.erase(clusterSeeds.begin()+indexCorePoint);

        for( vector<int>::size_type i = 0, n = clusterSeeds.size(); i < n; ++i )
        {
            vector<int> clusterNeighors = calculateCluster(m_boxes.at(clusterSeeds[i]));

            if ( clusterNeighors.size() >= m_minPoints )
            {
                vector<int>::iterator iterNeighors;
                for ( iterNeighors = clusterNeighors.begin(); iterNeighors != clusterNeighors.end(); ++iterNeighors )
                {
                    if ( m_boxes.at(*iterNeighors).clusterID == UNCLASSIFIED || m_boxes.at(*iterNeighors).clusterID == NOISE )
                    {
                        if ( m_boxes.at(*iterNeighors).clusterID == UNCLASSIFIED )
                        {
                            clusterSeeds.push_back(*iterNeighors);
                            n = clusterSeeds.size();
                        }
                        m_boxes.at(*iterNeighors).clusterID = clusterID;
                    }
                }
            }
        }

        return SUCCESS;
    }
}

vector<int> DBSCAN::calculateCluster(ClusterBBox point)
{
    int index = 0;
    vector<ClusterBBox>::iterator iter;
    vector<int> clusterIndex;
    for( iter = m_boxes.begin(); iter != m_boxes.end(); ++iter)
    {
        if ( calculateDistance(point, *iter) <= m_epsilon )
        {
            clusterIndex.push_back(index);
        }
        index++;
    }
    return clusterIndex;
}

vector<ClusterBBox> DBSCAN::getBoxes()
{
    return m_boxes;
}

inline double DBSCAN::calculateDistance(ClusterBBox pointCore, ClusterBBox pointTarget)
{
    int cLeft = pointCore.left;
    int cRight = pointCore.right;
    int cTop = pointCore.top;
    int cBottom = pointCore.bottom;
    int tLeft = pointTarget.left;
    int tRight = pointTarget.right;
    int tTop = pointTarget.top;
    int tBottom = pointTarget.bottom;

    // Check overlaps
    bool hOverlap = cRight >= tLeft && tLeft >= cLeft || tRight >= cLeft && cLeft >= tLeft;
    bool vOverlap = cBottom >= tTop && tTop >= cTop || tBottom >= cTop && cTop >= tTop;

    // Boxes touch
    if (hOverlap && vOverlap)
    {
        return 0;
    }
    // Vertical overlap (i.e. could be on same line)
    else if (vOverlap)
    {
        int dist1 = tLeft - cRight;
        int dist2 = cLeft - tRight;
        return max(dist1, dist2); 
    }
    // Rest of the boxes are not considered to be near enough
    else
    {
        return m_epsilon + 1;
    }
}