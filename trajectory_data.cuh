/*
 * trajectory_data.cuh
 *
 *  Created on: Sep 2, 2020
 *      Author: clark
 */

#ifndef TRAJECTORY_DATA_CUH_
#define TRAJECTORY_DATA_CUH_

// Includes
#include <vector>
#include <string>


// Structs

// Trajectory data will be organized as a vector of these points. This will make transfer of data to GPU possible.
// Trajectory level information will be maintained in a trajectory index if necessary
struct point{
	int trajectory_number;
	double x;
	double y;
	double t;
	__device__ __host__ point():trajectory_number(0),x(0.0),y(0.0),t(0.0){}
	__device__ __host__ point(int trajectory_number, double x, double y, double t):
			trajectory_number(trajectory_number), x(x), y(y), t(t){}
};

// Prototypes
int load_trajectory_data_from_file(std::string file_name, std::vector<point> &host_trajectory_data);

// TODO add SPR-Tree Trajectory index as described in Deng2015.

#endif /* TRAJECTORY_DATA_CUH_ */
