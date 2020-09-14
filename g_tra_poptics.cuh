/*
 * g_tra_poptics.cuh
 *
 *  Created on: Sep 2, 2020
 *      Author: clark
 */

#ifndef G_TRA_POPITCS_CUH_
#define G_TRA_POPITCS_CUH_

#include <vector>
#include "trajectory_data.cuh"
#include "strtree.cuh"

int g_tra_poptics(strtree strtree, int cpu_threads, double epsilon, double epsilon_prime, int min_num_trajectories);

struct test
{
		int boy1;
		int boy2;
};

#endif /* G_TRA_POPITCS_CUH_ */
