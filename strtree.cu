/*
 * strtree.cu
 *
 *  Created on: Sep 13, 2020
 *      Author: Clark Barrus based on rtree code from github.com/phifaner/comdb/rtree.cu
 */

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <iostream>
#include "strtree.cuh"

#define DEBUG true

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

//__host__ __device__
//int overlap(RTree_Rect *R, RTree_Rect * S)
//{
//    register RTree_Rect *r = R, *s = S;
//
//    assert(r && s);
//
//
//    if ( r->left > s->right || r->right < s->left
//           || r->top > s->bottom || r->bottom < s->top )
//    {
//
//    //printf("overlap R: %llu, %llu, %lu, %lu, S: %llu, %llu, %lu, %lu\n",
//    //        r->left, r->right, r->top, r->bottom, s->left, s->right, s->top, s->bottom);
//        return 0;
//    }
//    else
//        return 1;
//}

//__host__ __device__
//int contains(RTree_Rect *R, RTree_Point *P)
//{
//    register RTree_Rect *r = R;
//    register RTree_Point *p = P;
//
//    assert(r && p);
//
//    //printf("point: %llu, %lu, Rect: %llu, %llu, %lu, %lu\n",
//    //        p->x, p->y, r->left, r->right, r->top, r->bottom);
//
//    if (p->x < r->right && p->x > r->left
//            && p->y < r->bottom && p->y > r->top)
//        return 1;
//    else
//        return 0;
//}

__host__ __device__
inline void init_boundary(strtree_rect *bbox)
{
    bbox->x1 = DBL_MAX;
    bbox->x2 = -DBL_MAX;
    bbox->y1 = DBL_MAX;
    bbox->y2 = -DBL_MAX;
    bbox->t1 = DBL_MAX;
    bbox->t2 = -DBL_MAX;
}

//__host__ __device__
//inline void update_boundary(RTree_Rect *bbox, RTree_Rect *node_bbx)
//{
//    bbox->top = min(bbox->top, node_bbx->top);
//    bbox->bottom = max(bbox->bottom, node_bbx->bottom);
//    bbox->left = min(bbox->left, node_bbx->left);
//    bbox->right = max(bbox->right, node_bbx->right);
//
//    //printf("---node bbox: %llu, %llu, update: %llu, %llu\n",
//    //        node_bbx->left, node_bbx->right, bbox->left, bbox->right);
//
//}

__host__ __device__
inline void update_boundary(strtree_rect *bbox, strtree_rect *node_bbx)
{
	bbox->x1 = min(node_bbx->x1, bbox->x1);
	bbox->x2 = max(node_bbx->x2, bbox->x2);
	bbox->y1 = min(node_bbx->y1, bbox->y1);
	bbox->y2 = max(node_bbx->y2, bbox->y2);
	bbox->t1 = min(node_bbx->t1, bbox->t1);
	bbox->t2 = max(node_bbx->t2, bbox->t2);
}

__host__ __device__
inline void c_update_boundary(strtree_rect *bbox, strtree_line *p)
{
    bbox->x1 = min(p->line_boundingbox.x1, bbox->x1);
    bbox->x2 = max(p->line_boundingbox.x2, bbox->x2);
    bbox->y1 = min(p->line_boundingbox.y1, bbox->y1);
    bbox->y2 = max(p->line_boundingbox.y2, bbox->y2);
    bbox->t1 = min(p->line_boundingbox.t1, bbox->t1);
	bbox->t2 = max(p->line_boundingbox.t2, bbox->t2);

    //printf("x: %llu, bbox: %lu, %lu, %llu, %llu\n", p->x, bbox->top, bbox->bottom, bbox->left, bbox->right);
}

__host__ __device__
inline size_t get_node_length (
        const size_t i,
        const size_t len_level,
        const size_t previous_level_len,
        const size_t node_size)
{
    const size_t n = node_size;
    const size_t len = previous_level_len;
    const size_t final_i = len_level -1;

    // set lnum to len%n if it's the last iteration and there's a remainder, else n
    return ((i != final_i || len % n == 0) *n) + ((i == final_i && len % n != 0) * (len % n));
}

//// points are on device and sorted by x
//void cuda_sort(RTree_Points *sorted)
//{
//    uint64 *X = sorted->X;
//    unsigned long *Y = sorted->Y;
//    int *ID = sorted->ID;
//
//    // sort by x
//    auto tbegin = thrust::make_zip_iterator(thrust::make_tuple(Y, ID));
//    auto tend = thrust::make_zip_iterator(thrust::make_tuple(Y+sorted->length, ID+sorted->length));
//    thrust::sort_by_key(thrust::device, X, X+sorted->length, tbegin);
//
//}

host_strtree cuda_create_host_strtree(strtree_lines lines)
{
	// Skip sorting trajectories and lines for now. Not sure how to implement/how it would effect organization.
    //cuda_sort(&lines);

    strtree_leaf *leaves = cuda_create_leaves( &lines );

    const size_t len_leaf = DIV_CEIL(lines.length, STRTREE_NODE_SIZE);

//    // build rtree from bottom
    strtree_node *level_previous  = (strtree_node*) leaves;
    size_t      len_previous    = len_leaf;
    size_t      depth           = 1;    // leaf level: 0
    size_t      num_nodes       = len_leaf;
    while (len_previous > STRTREE_NODE_SIZE)
    {
        level_previous = cuda_create_level(level_previous, len_previous, depth);
        num_nodes += level_previous->num;
        len_previous = DIV_CEIL(len_previous, STRTREE_NODE_SIZE);
        ++depth;
    }

    // tackle the root node
    strtree_node *root = new strtree_node();
    init_boundary(&root->boundingbox);
    root->num = len_previous;
    root->children = level_previous;
    num_nodes += root->num;
    for (size_t i = 0, end = len_previous; i != end; ++i)
        update_boundary(&root->boundingbox, &root->children[i].boundingbox);
    ++depth;
    root->depth = depth;

    host_strtree tree = {depth, root};
    //    size_t      len_previous    = len_leaf;
    //    size_t      depth           = 1;    // leaf level: 0
    //    size_t      num_nodes       = len_leaf;
    //    while (len_previous > STRTREE_NODE_SIZE)
    //    {
    //        level_previous = cuda_create_level(level_previous, len_previous, depth);
    //        num_nodes += level_previous->num;
    //        len_previous = DIV_CEIL(len_previous, STRTREE_NODE_SIZE);
    //        ++depth;
    //    }

    if (DEBUG)
	{
		std::cout << "Root node at level " << depth << "create_strtree() returns:" << std::endl;
		strtree_node node = *root;
		std::cout << "Root node: num=" << node.num << ": depth=" << node.depth
				<< ": bbox.x1=" << node.boundingbox.x1 << ": bbox.x2=" << node.boundingbox.x2
				<< ": bbox.y1=" << node.boundingbox.y1 << ": bbox.y2=" << node.boundingbox.y2
				<< ": bbox.t1=" << node.boundingbox.t1 << ": bbox.t2=" << node.boundingbox.t2 << std::endl;
		for (int j = 0; j < node.num; j++)
		{
			strtree_node child_node = node.children[j];
			std::cout << "    Child node " << j << ": num=" << child_node.num << ", depth=" << child_node.depth
				<< ", bbox.x1=" << child_node.boundingbox.x1 << ", bbox.x2=" << child_node.boundingbox.x2
				<< ", bbox.y1=" << child_node.boundingbox.y1 << ", bbox.y2=" << child_node.boundingbox.y2
				<< ", bbox.t1=" << child_node.boundingbox.t1 << ", bbox.t2=" << child_node.boundingbox.t2
				<< std::endl;
		}
	}

    return tree;
}


strtree cuda_create_strtree(thrust::host_vector<strtree_line> h_lines)
{
	// Skip sorting trajectories and lines for now. Not sure how to implement/how it would effect organization.
    //cuda_sort(&lines);

//    strtree_leaf *leaves = cuda_create_leaves( &lines );

//    const size_t len_leaf = DIV_CEIL(lines.length, STRTREE_NODE_SIZE);

    thrust::device_vector<strtree_line> d_lines = h_lines;
    thrust::device_vector<strtree_node> d_nodes(2);

//    // build rtree from bottom
//    strtree_node *level_previous  = (strtree_node*) leaves;
//    size_t      len_previous    = len_leaf;
//    size_t      depth           = 1;    // leaf level: 0
//    size_t      num_nodes       = len_leaf;
//    while (len_previous > STRTREE_NODE_SIZE)
//    {
//        level_previous = cuda_create_level(level_previous, len_previous, depth);
//        num_nodes += level_previous->num;
//        len_previous = DIV_CEIL(len_previous, STRTREE_NODE_SIZE);
//        ++depth;
//    }

    // tackle the root node
    size_t root_offset = 0;
//    init_boundary(&root->boundingbox);
//    root->num = len_previous;
//    root->children = level_previous;
//    num_nodes += root->num;
//    for (size_t i = 0, end = len_previous; i != end; ++i)
//        update_boundary(&root->boundingbox, &root->children[i].boundingbox);
//    ++depth;
//    root->depth = depth;

    strtree tree = {root_offset, d_nodes, d_lines};
//
//    if (DEBUG)
//	{
//		std::cout << "Root node at level " << depth << "create_strtree() returns:" << std::endl;
//		strtree_node node = *root;
//		std::cout << "Root node: num=" << node.num << ": depth=" << node.depth
//				<< ": bbox.x1=" << node.boundingbox.x1 << ": bbox.x2=" << node.boundingbox.x2
//				<< ": bbox.y1=" << node.boundingbox.y1 << ": bbox.y2=" << node.boundingbox.y2
//				<< ": bbox.t1=" << node.boundingbox.t1 << ": bbox.t2=" << node.boundingbox.t2 << std::endl;
//		for (int j = 0; j < node.num; j++)
//		{
//			strtree_node child_node = node.children[j];
//			std::cout << "    Child node " << j << ": num=" << child_node.num << ", depth=" << child_node.depth
//				<< ", bbox.x1=" << child_node.boundingbox.x1 << ", bbox.x2=" << child_node.boundingbox.x2
//				<< ", bbox.y1=" << child_node.boundingbox.y1 << ", bbox.y2=" << child_node.boundingbox.y2
//				<< ", bbox.t1=" << child_node.boundingbox.t1 << ", bbox.t2=" << child_node.boundingbox.t2
//				<< std::endl;
//		}
//	}

    return tree;
}

__global__
void create_level_kernel
        (
            strtree_node *next_level,
            strtree_node *nodes,
            strtree_node *real_nodes,
            const size_t len,
            size_t depth
         )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t next_level_len = DIV_CEIL(len, STRTREE_NODE_SIZE);

    if (i >= next_level_len) return;    // skip the final block remainder

    strtree_node *n = &next_level[i];
    init_boundary(&n->boundingbox);
    n->num = get_node_length(i, next_level_len, len, STRTREE_NODE_SIZE);
    n->children = &real_nodes[i * STRTREE_NODE_SIZE]; // Save host reference to child nodes
    n->depth = depth;
    //printf("level num: %d, ---num: %lu\n", n->num, next_level_len);


	#pragma unroll
    for (size_t j = 0, jend = n->num; j != jend; ++j)
    {
        update_boundary(&n->boundingbox, &nodes[i * STRTREE_NODE_SIZE + j].boundingbox); // Use device reference to actually access child node
        //printf("after set node bbox: %lu, %lu, %llu, %llu\n",
        //    n->bbox.top, n->bbox.bottom, n->bbox.left, n->bbox.right);
    }
}

strtree_node* cuda_create_level(strtree_node *nodes, const size_t len, size_t depth)
{
    //Should be set somewhere else. const size_t THREADS_PER_BLOCK = 512;
    const size_t next_level_len = DIV_CEIL(len, STRTREE_NODE_SIZE);

    strtree_node *d_nodes;
    strtree_node *d_next_level;
    cudaMalloc( (void**) &d_nodes, len * sizeof(strtree_node) );
    cudaMalloc( (void**) &d_next_level, next_level_len * sizeof(strtree_node) );

    cudaMemcpy(d_nodes, nodes, len * sizeof(strtree_node), cudaMemcpyHostToDevice);

    create_level_kernel<<< (next_level_len + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
            (d_next_level, d_nodes, nodes, len, depth);

    strtree_node *next_level = new strtree_node[next_level_len];
    cudaMemcpy(next_level, d_next_level, next_level_len * sizeof(strtree_node), cudaMemcpyDeviceToHost);

    cudaFree(d_next_level);
    cudaFree(d_nodes);

    if (DEBUG)
	{
		std::cout << "Level " << depth << " nodes generated on device after transfer back to host, cuda_create_level() returns:" << std::endl;
		for (int i = 0; i < next_level_len; i++) {
			strtree_node node = next_level[i];
			std::cout << "Node: " << i << ": num=" << node.num << ": depth=" << node.depth
					<< ": bbox.x1=" << node.boundingbox.x1 << ": bbox.x2=" << node.boundingbox.x2
					<< ": bbox.y1=" << node.boundingbox.y1 << ": bbox.y2=" << node.boundingbox.y2
					<< ": bbox.t1=" << node.boundingbox.t1 << ": bbox.t2=" << node.boundingbox.t2 << std::endl;
			for (int j = 0; j < node.num; j++)
			{
				strtree_node child_node = node.children[j];
				std::cout << "    Child node " << j << ": num=" << child_node.num << ", depth=" << child_node.depth
					<< ", bbox.x1=" << child_node.boundingbox.x1 << ", bbox.x2=" << child_node.boundingbox.x2
					<< ", bbox.y1=" << child_node.boundingbox.y1 << ", bbox.y2=" << child_node.boundingbox.y2
					<< ", bbox.t1=" << child_node.boundingbox.t1 << ", bbox.t2=" << child_node.boundingbox.t2
					<< std::endl;
			}
		}
	}

    return next_level;
}

// Each thread populates an assigned leaf with entries
__global__
void create_leaves_kernel
        (
            strtree_leaf    *leaves,
            strtree_line    *lines,
            strtree_line    *h_lines,
            int				*ID,
            int				*Trajectory_Number,
            strtree_rect	*Line_BoundingBox,
            short			*Orientation,
            size_t			len
        )
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    const size_t len_leaf = DIV_CEIL(len, STRTREE_NODE_SIZE); // Total number of leaf nodes
    if (i >= len_leaf) return;  // skip the final block remainder

    // tackle leaf lines
    strtree_leaf *l = &leaves[i]; // Leaf assigned to this thread
    init_boundary(&l->boundingbox);
    l->num = get_node_length(i, len_leaf, len, STRTREE_NODE_SIZE);
    l->depth = 0;
    l->lines = &h_lines[i * STRTREE_NODE_SIZE]; // occupy position

    // compute MBR from each point in the node
	#pragma unroll
    for (size_t j = 0, jend = l->num; j != jend; ++j)
    {
//        // *** use pointer, not value ***/ // *** I am using value, not sure how to use pointer and have update boundary work? ***/
    	strtree_line *line 		= &lines[i * STRTREE_NODE_SIZE + j];
        line->id				= ID[i * STRTREE_NODE_SIZE + j];
        line->line_boundingbox	= Line_BoundingBox[i * STRTREE_NODE_SIZE + j];
        line->orientation 		= Orientation[i * STRTREE_NODE_SIZE + j];
        line->trajectory_number = Trajectory_Number[i * STRTREE_NODE_SIZE + j];
////        p->x     = X[i   * STRTREE_NODE_SIZE + j];
////        p->y     = Y[i   * STRTREE_NODE_SIZE + j];
////        p->id    = ID[i  * STRTREE_NODE_SIZE + j];
//
//        //printf("----------id: %d, j: %lu\n", line->id, j);
        c_update_boundary(&l->boundingbox, line);
    }
}

strtree_leaf* cuda_create_leaves(strtree_lines *sorted_lines)
{
    const size_t len = sorted_lines->length;
    const size_t num_leaf = DIV_CEIL(len, STRTREE_NODE_SIZE);

    strtree_leaf  *d_leaves;
    strtree_line  *d_lines;
    int *d_ID;
    int *d_Trajectory_Number;
    strtree_rect *d_Line_BoundingBox;
    short *d_Orientation;

    if (DEBUG) { std::cout <<"Starting cudaMalloc for creating leaves" << std::endl; }

    cudaMalloc( (void**) &d_leaves, num_leaf    * sizeof(strtree_leaf) );
    cudaMalloc( (void**) &d_lines, len          * sizeof(strtree_line) );

    if (DEBUG) { std::cout <<"Starting cudaMalloc for sorted_lines contents" << std::endl; }

    // Move a copy of sorted_lines to the device so the device can see the data
    if (DEBUG) { std::cout <<"Starting cudaMalloc for d_ID" << std::endl; }
    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_ID, len * sizeof(int)));
    if (DEBUG) { std::cout <<"Starting cudaMemcpy for d_ID" << std::endl; }
    CUDA_CHECK_RETURN(cudaMemcpy(d_ID, sorted_lines->ID, len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Trajectory_Number, len * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_Trajectory_Number, sorted_lines->Trajectory_Number, len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Line_BoundingBox, len * sizeof(strtree_rect)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_Line_BoundingBox, sorted_lines->Line_BoundingBox, len * sizeof(strtree_rect), cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Orientation, len * sizeof(short)));
    CUDA_CHECK_RETURN(cudaMemcpy(d_Orientation, sorted_lines->Orientation, len * sizeof(short), cudaMemcpyHostToDevice));
	//d_sorted_lines->length = sorted_lines->length;

    if (DEBUG) { std::cout <<"Finished cudaMalloc and memcpy for sorted_lines sorted_lines contents" << std::endl; }

    // Leaves on device will copy lines into here this host array and maintain pointers to positions in this array
    strtree_line *lines = new strtree_line[len];

    if (DEBUG) { std::cout <<"Launching create_leaves_kernel" << std::endl; }

    create_leaves_kernel<<< (num_leaf + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (d_leaves, d_lines, lines, d_ID, d_Trajectory_Number, d_Line_BoundingBox, d_Orientation, sorted_lines->length);

    if (DEBUG) { std::cout <<"Finished create_leaves_kernel" << std::endl; }

    strtree_leaf  *leaves = new strtree_leaf[num_leaf];

    // copy points from device to host
    cudaMemcpy(leaves, d_leaves, num_leaf   * sizeof(strtree_leaf), cudaMemcpyDeviceToHost);
    cudaMemcpy(lines, d_lines, len        * sizeof(strtree_line), cudaMemcpyDeviceToHost);

    cudaFree(d_leaves);
    cudaFree(d_lines);
    cudaFree(d_ID);
    cudaFree(d_Trajectory_Number);
    cudaFree(d_Line_BoundingBox);
    cudaFree(d_Orientation);

    if (DEBUG)
	{
    	std::cout << "Lines copied from device point to the following lines now on host" << std::endl;
		for(int i = 0; i < len; i++)
		{
			strtree_line line = lines[i];
			std::cout << "Line " << i << ": id=" << line.id << ", trajectory_number=" << line.trajectory_number
					<< ", bbox.x1=" << line.line_boundingbox.x1 << ", bbox.x2=" << line.line_boundingbox.x2
					<< ", bbox.y1=" << line.line_boundingbox.y1 << ", bbox.y2=" << line.line_boundingbox.y2
					<< ", bbox.t1=" << line.line_boundingbox.t1 << ", bbox.t2=" << line.line_boundingbox.t2
					<< ", orientation=" << line.orientation << std::endl;
		}
		std::cout << "Leaves generated on device after transfer back to host:" << std::endl;
		for (int i = 0; i < num_leaf; i++) {
			strtree_leaf leaf = leaves[i];
			std::cout << "Leaf " << i << ": num=" << leaf.num << ": depth=" << leaf.depth
					<< ": bbox.x1=" << leaf.boundingbox.x1 << ": bbox.x2=" << leaf.boundingbox.x2
					<< ": bbox.y1=" << leaf.boundingbox.y1 << ": bbox.y2=" << leaf.boundingbox.y2
					<< ": bbox.t1=" << leaf.boundingbox.t1 << ": bbox.t2=" << leaf.boundingbox.t2 << std::endl;
    		for (int j = 0; j < leaf.num; j++)
    		{
    			strtree_line line = leaf.lines[j];
    			std::cout << "    Child line " << j << ": id=" << line.id << ", trajectory_number=" << line.trajectory_number
					<< ", bbox.x1=" << line.line_boundingbox.x1 << ", bbox.x2=" << line.line_boundingbox.x2
					<< ", bbox.y1=" << line.line_boundingbox.y1 << ", bbox.y2=" << line.line_boundingbox.y2
					<< ", bbox.t1=" << line.line_boundingbox.t1 << ", bbox.t2=" << line.line_boundingbox.t2
					<< ", orientation=" << line.orientation << std::endl;
    		}
		}
	}

    return leaves;

}
//
//int cpu_search(RTree_Node *N, RTree_Rect *rect, std::vector<int> &points)
//{
//    register RTree_Node *n = N;
//    register RTree_Rect *r = rect;
//    register int hit_count = 0;
//    register int i;
//
//    assert(n);
//    assert(n->num);
//    assert(r);
//
//    //printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\t rect: %llu, %lu\n", n->depth, n->bbox.left, n->bbox.right,
//    //     n->bbox.top, n->bbox.bottom, r->left, r->top );
//
//
//    if (n->depth > 0)
//    {
//        for (i = 0; i < n->num; i++)
//        {
//            printf("depth: %lu, bbox: %llu, %llu, %lu, %lu\t rect: %llu, %lu\t num: %lu\n",
//                    n->depth, n->children[i].bbox.left, n->children[i].bbox.right,
//                        n->children[i].bbox.top, n->children[i].bbox.bottom, r->left, r->top
//                        , n->children[i].num );
//
//            if ( overlap(r, &n->children[i].bbox) )
//            {
//                hit_count += cpu_search(&n->children[i], rect, points);
//            }
//        }
//    }
//    else    // this is a leaf node
//    {
//        if ( n->num && overlap(r, &n->bbox) )
//        {
//
//            //printf("---%llu, %llu, %lu, %lu\n", n->bbox.left, n->bbox.right, n->bbox.top, n->bbox.bottom);
//
//            RTree_Leaf *l = (RTree_Leaf*) n;
//            for (i = 0; i < n->num; i++)
//            {
//                // determine whether points in rect
//                if ( contains(r, &l->points[i] ) )
//                {
//                    hit_count++;
//
//                    // check if contains this point
//                    if ( std::find(points.begin(), points.end(), l->points[i].id) == points.end() )
//                        points.push_back(l->points[i].id);
//
//                    printf("%d trajectory is hit, %llu, %lu\n", l->points[i].id, l->points[i].x, l->points[i].y);
//                }
//            }
//        }
//    }
//
//    return hit_count;
//}
//
//template< int MAX_THREADS_PER_BLOCK >
//__global__
//void search_kernel(
//        CUDA_RTree_Node     *   d_nodes,
//        int                 *   d_edges,
//        RTree_Rect          *   d_rects,
//        bool                *   d_search_front,
//        RTree_Rect          *   rects,
//        int                 *   results,
//        int                     num_nodes)
//{
//    // shared memory to store the query rectangles
//    extern __shared__ RTree_Rect rmem[];
//
//    // Address of shared memory
//    RTree_Rect *s_rect = (RTree_Rect *) &rmem[blockIdx.x];
//
//    // each thread represents one node
//    int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
//
//    // whether the query rectangle overlaps the MBR of the frontier node
//    bool flag = false;
//    if ( overlap(&d_rects[tid], s_rect) ) flag = true;
//
//    // node is in frontier and its MBR overlaps query rectangle
//    if (tid < num_nodes && d_search_front[tid] && flag)
//    {
//        // remove it from frontier
//        d_search_front[tid] = false;
//
//        // reach Leaf level
//        if (d_nodes[tid].starting == -1)
//        {
//            results[tid] = 1;
//            return ;
//        }
//
//        // put its children to the next search_front
//        for (int i = d_nodes[tid].starting; i < (d_nodes[tid].num_edges + d_nodes[tid].starting); i++)
//        {
//            int id = d_edges[i];
//            d_search_front[id] = true;
//        }
//    }
//
//    search_kernel<MAX_THREADS_PER_BLOCK><<<10, 20>>>
//        (d_nodes, d_edges, d_rects, d_search_front, rects,  results, num_nodes);
//
//}
//
//void fill_edges(RTree_Node *N, CUDA_RTree_Node *h_nodes, int *h_edges, RTree_Rect *h_rects, int& node_id)
//{
//    register RTree_Node * n = N;
//
//    if (node_id == 0)
//    {
//        h_nodes[node_id].starting = 0;  // initialize root node
//
//        for (int i = h_nodes[0].starting; i < (h_nodes[0].starting + n->num); i++)
//        {
//
//            // starting index of child in array
//            if (i == 0)
//                h_edges[i] = STRTREE_NODE_SIZE;
//            else
//                h_edges[i] = n->num;
//
//        }
//    }
//    else
//    {
//
//        if (n->depth > 0) // set nodes
//        {
//            h_nodes[node_id].starting = h_nodes[node_id-1].starting + h_nodes[node_id-1].num_edges;
//
//            for (int i = h_nodes[node_id].starting; i < (h_nodes[node_id].starting + n->num); i++)
//            {
//                // starting index of child in array
//                h_edges[i] = h_edges[i-1] + h_nodes[node_id-1].num_edges;
//
//            }
//        }
//        else    // set Leaf node
//        {
//            h_nodes[node_id].starting = -1;
//        }
//    }
//
//    h_nodes[node_id].num_edges = n->num;
//    h_rects[node_id] = n->bbox;
//
//    // recursively fill edges
//    for (int i = 0; i < n->num; i++)
//    {
//        fill_edges(&n->children[i], h_nodes, h_edges, h_rects, ++node_id);
//    }
//
//}

//RTree_Points cuda_search(RTree *tree, std::vector<RTree_Rect> rect_vec)
//{
//    CUDA_RTree_Node *   h_nodes = (CUDA_RTree_Node *) malloc(tree->num * sizeof(CUDA_RTree_Node));
//    int *               h_edges = (int *) malloc(tree->num * sizeof(int) * STRTREE_NODE_SIZE);
//    RTree_Rect      *   h_rects = (RTree_Rect *) malloc(tree->num * sizeof(RTree_Rect));
//
//    int node_id = 0;
//
//
//    printf("tree node number: %lu-----\n", tree->num);
//
//    // copy data from cpu to gpu
//    fill_edges(tree->root, h_nodes, h_edges, h_rects, node_id);
//
//    for (int i = 0; i < tree->num; i++)
//    {
//        printf("starting of node: %d is %d\n", i, h_nodes[i].starting);
//    }
//    // allocate n blocks to deal with n query rectangles
//
//    RTree_Points points;
//
//    return points;
//}

strtree_lines points_to_lines(point* points, trajectory_index* trajectory_indices, int num_points, int num_trajectories)
{
	// Want to create the following struct given information from a file.
	//	struct strtree_lines
	//	{
	//		int 		*ID;
	//		int 		*Trajectory_Number;
	//
	//		strtree_rect *Line_BoundingBox;
	//
	//		short *Orientation;
	//
	//		size_t length;
	//	};

	// Each trajectory has 1 fewer lines than points.
	size_t num_lines = num_points - num_trajectories;

	int *ID = new int[num_lines];
	int *cur_ID = ID;
	int id = 0;
	int *Trajectory_Number = new int[num_lines];
	int *cur_Trajectory_Number = Trajectory_Number;
	strtree_rect *Line_BoundingBox = new strtree_rect[num_lines];
	strtree_rect *cur_Line_BoundingBox = Line_BoundingBox;
	short *Orientation = new short[num_lines];
	short *cur_Orientation = Orientation;

	point last_point = points[0];
	point this_point;
	for (int i = 1; i < num_points; i++) //TODO do this work using a GPU kernel
	{
		this_point = points[i];

		if(DEBUG)
		{
			std::cout << "Adding line between points:" << std::endl;
			std::cout << "Traj_num: " << last_point.trajectory_number << " x: " << last_point.x << " y: " << last_point.y << " t: " << last_point.t << std::endl;
			std::cout << "Traj_num: " << this_point.trajectory_number << " x: " << this_point.x << " y: " << this_point.y << " t: " << this_point.t << std::endl;
		}

		if (last_point.trajectory_number != this_point.trajectory_number)
		{
			// We are now on a new trajectory. Don't add another line
			last_point = this_point;
			continue;
		}

		// Create a line between last point and this point.
		*cur_ID = id;
		*cur_Trajectory_Number = this_point.trajectory_number;
		*cur_Line_BoundingBox = points_to_bbox(last_point, this_point);
		*cur_Orientation = points_to_orientation(last_point, this_point);

		// Set up for next execution of the for loop
		cur_ID++;
		id++;
		cur_Trajectory_Number++;
		cur_Line_BoundingBox++;
		cur_Orientation++;
		last_point = this_point;
	}

	if (DEBUG)
	{
		std::cout << "Contents of strtree_lines lines returned by points_to_lines()" << std::endl;
		for(int i = 0; i < num_lines; i++)
		{
			std::cout << "Line " << i << ": id=" << ID[i] << ", trajectory_number=" << Trajectory_Number[i]
					<< ", bbox.x1=" << Line_BoundingBox[i].x1 << ", bbox.x2=" << Line_BoundingBox[i].x2
					<< ", bbox.y1=" << Line_BoundingBox[i].y1 << ", bbox.y2=" << Line_BoundingBox[i].y2
					<< ", bbox.t1=" << Line_BoundingBox[i].t1 << ", bbox.t2=" << Line_BoundingBox[i].t2
					<< ", orientation=" << Orientation[i] << std::endl;
		}
	}

	strtree_lines lines = {ID, Trajectory_Number, Line_BoundingBox, Orientation, num_lines};
	return lines;
}

// V2.0 of points to lines. This is the iteration of the data structure using offsets and vector instead of arrays & pointers.
thrust::host_vector<strtree_line> points_to_line_vector(
		point* points, trajectory_index* trajectory_indices, int num_points, int num_trajectories)
{
	// Want to create an vector of strtree_line structs
	//	std::vector<strtree_line> lines

	// Each trajectory has 1 fewer lines than points.
	size_t num_lines = num_points - num_trajectories;

	thrust::host_vector<strtree_line> lines(num_lines);
	int id = 0; // Use line id as an index tracking which line we are on.

	point last_point = points[0];
	point this_point;
	for (int i = 1; i < num_points; i++) //TODO do this work using a GPU kernel
	{
		this_point = points[i];

		if(DEBUG)
		{
			std::cout << "Adding line between points:" << std::endl;
			std::cout << "Traj_num: " << last_point.trajectory_number << " x: " << last_point.x << " y: " << last_point.y << " t: " << last_point.t << std::endl;
			std::cout << "Traj_num: " << this_point.trajectory_number << " x: " << this_point.x << " y: " << this_point.y << " t: " << this_point.t << std::endl;
		}

		if (last_point.trajectory_number != this_point.trajectory_number)
		{
			// We are now on a new trajectory. Don't add another line
			last_point = this_point;
			continue;
		}

		strtree_line *line = &lines[id];

		// Create a line between last point and this point.
		line->id = id;
		line->trajectory_number = this_point.trajectory_number;
		line->line_boundingbox = points_to_bbox(last_point, this_point);
		line->orientation = points_to_orientation(last_point, this_point);

		// Set up for next execution of the for loop
		id++;
		last_point = this_point;
	}

	if (DEBUG)
	{
		std::cout << "Contents of vector<strtree_line> lines returned by points_to_line_vector()" << std::endl;
		for(int i = 0; i < num_lines; i++)
		{
			std::cout << "Line " << i << ": id=" << lines[i].id << ", trajectory_number=" << lines[i].trajectory_number
					<< ", bbox.x1=" << lines[i].line_boundingbox.x1 << ", bbox.x2=" << lines[i].line_boundingbox.x2
					<< ", bbox.y1=" << lines[i].line_boundingbox.y1 << ", bbox.y2=" << lines[i].line_boundingbox.y2
					<< ", bbox.t1=" << lines[i].line_boundingbox.t1 << ", bbox.t2=" << lines[i].line_boundingbox.t2
					<< ", orientation=" << lines[i].orientation << std::endl;
		}
	}

	return lines;
}

strtree_rect points_to_bbox(point p1, point p2)
{
	strtree_rect rect;
	rect.x1 = min(p1.x, p2.x);
	rect.x2 = max(p1.x, p2.x);
	rect.y1 = min(p1.y, p2.y);
	rect.y2 = max(p1.y, p2.y);
	rect.t1 = min(p1.t, p2.t);
	rect.t2 = max(p1.t, p2.t);
	return rect;
}

short points_to_orientation(point p1, point p2)
{
	/** Recall:
	 * orientation indicates how the line represented by the entry is represented by the bounding box above
	 * Since trajectories move forward in time, first point is always t1, second t2.
	 * 0: (x1,y1) to (x2,y2)
	 * 1: (x1,y2) to (x2,y1)
	 * 2: (x2,y1) to (x1,y2)
	 * 3: (x2,y2) to (x1,y1)
	 */
	if (p1.x < p2.x)
	{
		if (p1.y < p2.y)
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
	else
	{
		if (p1.y < p2.y)
		{
			return 2;
		}
		else
		{
			return 3;
		}
	}
}
