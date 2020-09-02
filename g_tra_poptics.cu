/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "g_tra_poptics.cuh"

int g_tra_poptics(std::vector<point> host_trajectory_data, int cpu_threads,
		double epsilon, double epsilon_prime, int min_num_trajectories)
{
	// Load trajectory data onto GPU

	// Divide data into subsets (uses either CPU multithreading OR GPU dynamic concurrency

	// Initialize global priority

	// For each CPU thread generate local MST's (need GPU version)

	// For each trajectory in data set
	// Mark as processed
	// Get neighbors of trajectory (TODO)
	// set_core_distance(tr, Ns, epsilon, min num of trs)
	// if tr.coreDistance != null
	// update local priority queue with edges to each neighbor
	// Until local priority queue empty
	// Process next edge in priority queue

	// Process next edge in priority queue:
	// Take minimum edge. Insert into global priority queue.
	// set core distance for this trajectory
	//

	// Find neighbors from global dataset
	// Add neighbors to local priority queue
	// Repeat until priority queue is empty

	return 0;
}
