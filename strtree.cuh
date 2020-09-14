/*
 * strtree.cuh
 *
 *  Created on: Sep 13, 2020
 *      Author: Clark Barrus, based on code from github.com/phifaner/comdb/rtree.h
 */

#ifndef STRTREE_CUH_
#define STRTREE_CUH_

#include <stdlib.h>
#include <vector>
#include "trajectory_data.cuh"

//#include "s2/base/integral_types.h"
typedef unsigned long long  uint64; // This is the only type used by the above include.

#define DIV_CEIL(x, y) (((x) + (y) - 1) / (y))

const size_t THREADS_PER_BLOCK = 512;

struct strtree_rect
{
    double x1, x2;
    double y1, y2;
    double t1, t2;
};

/**
 * STRTree_Point represents a single entry in an STR tree and is actually a line.
 * The entry contains a bounding box for the line and the orientation of the bounding box in space.
 */
struct strtree_line
{
	// ID number of the specific point. May not be necessary
	int 		id;
	// Point belongs to trajectory with this number
	int 		trajectory_number;

	// Bounding box made of 2 three-dimensional points
    strtree_rect line_boundingbox;

    /**
     * orientation indicates how the line represented by the entry is represented by the bounding box above
     * Since trajectories move forward in time, first point is always t1, second t2.
     * 0: (x1,y1) to (x2,y2)
     * 1: (x1,y2) to (x2,y1)
     * 2: (x2,y1) to (x1,y2)
     * 3: (x2,y2) to (x1,y1)
     */
    short orientation;

    bool operator==(strtree_line line)
    {
        if (line.id == id) return true;
        return false;
    }
};

struct strtree_lines
{
	int 		*ID;
	int 		*Trajectory_Number;

	strtree_rect *Line_BoundingBox;

	short *Orientation;

	size_t length;
};

#define STRTREE_NODE_SIZE         2
//#define MAX_THREADS_PER_BLOCK   100

struct strtree_node;
struct strtree
{
    size_t          depth;   	// know where the leaves begin
    strtree_node   *  root; 	// First node in the str tree
    size_t          num_nodes;	// Number of nodes in str tree
};



struct strtree_leaf
{
    strtree_rect    boundingbox;
    size_t          num;        // number of children
    size_t          depth;      // level
    strtree_line *  lines;		// line entries in leaf node

};

struct strtree_node
{
    strtree_rect      boundingbox;
    size_t          num;		// number of children
    size_t          depth;		// node level
    strtree_node *    children; 	// pointers to childen nodes
};

//struct CUDA_RTree_Node
//{
//    int starting;
//    int num_edges;
//};

//struct CUDA_RTree_Node *d_nodes;
//int *d_edeges;
//struct RTree_Rect *d_rects;

// Cuda sort algorithm to do low-x bulk loading
void cuda_sort(strtree_lines *lines);

// Creates an STR tree from lines using a Cuda low-x1 bulk loading procedure
strtree cuda_create_strtree(strtree_lines lines);

// Helper methods for low-x1 bulk loading procedure
strtree_leaf* cuda_create_leaves(strtree_lines *sorted);
strtree_node* cuda_create_level(strtree_node *nodes, const size_t len, size_t depth);

// Creates an STR tree serially using the insert method found in Pfoser2000 STR Tree paper
strtree serial_create_strtree(strtree_lines lines);

// Helper methods for serial strtree construction
void insert();
void split();
void findnode();
void chooseleaf();
void quadraticsplit();
void adjusttree();

strtree_lines points_to_lines(point* points, trajectory_index* trajectory_indices, int num_points, int num_trajectories);

// Helper methods for points_to_lines
strtree_rect points_to_bbox(point p1, point p2);
short points_to_orientation(point p1, point p2);


//int cpu_search(RTree_Node *N, RTree_Rect *rect, std::vector<int> &points);
//
//RTree_Points cuda_search(RTree *tree, std::vector<RTree_Rect> rect_vec);




#endif /* STRTREE_CUH_ */
