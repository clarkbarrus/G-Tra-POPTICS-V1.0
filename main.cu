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

#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

#include "trajectory_data.cuh"
#include "g_tra_poptics.cuh"
#include "rtree.h"
#include "strtree.cuh"

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

/**
 *
 * Entry point for executing G-Tra-POPTICS
 */
int main(int argc, char **argv)
{

	std::string file_name = "testtrajectorydata.csv";

	// Load trajectory data from file
	file_trajectory_data trajectory_data = load_trajectory_data_from_file(file_name);

	// Preprocessing: build STR-tree index
	strtree_lines lines = points_to_lines(trajectory_data.points, trajectory_data.trajectories,
			trajectory_data.num_points, trajectory_data.num_trajectories);
	host_strtree strtree = cuda_create_host_strtree(lines);

//	/* Initialize variables for G-Tra-POPTICS execution */
//	// Number of CPU threads executing
//	int cpu_threads = 8;
//	// Maximum epsilon at which clusters are detected
//	double epsilon = 0.2;
//	// Specific epsilon for which to find clusters after minimum spanning trees are built
//	double epsilon_prime = 0.1;
//	// Minimum number of trajectories near a point for it to be considered a core point.
//	double min_num_trajectories = 2;
//
//	// Execute G-Tra-POPTICS on data file
//	g_tra_poptics(strtree, cpu_threads, epsilon, epsilon_prime, min_num_trajectories);

	return 0;
}

