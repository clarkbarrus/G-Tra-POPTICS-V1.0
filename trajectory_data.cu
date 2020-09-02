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
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "trajectory_data.cuh"

#define DEBUG true

int load_trajectory_data_from_file(std::string file_name, std::vector<point> &host_trajectory_data)
{

	if (DEBUG)
	{
		std::cout << "Loading data from " << file_name << std::endl;
	}

	// TODO Should first count the number of trajectories and number of points to allow for linear time vector construction

	std::ifstream input_file;
	input_file.open(file_name);

	// Error check input file stream
	if (!input_file.is_open())
	{
		std::cerr << "Input file unable to open" << std::endl;
	}

	std::string line;

	getline(input_file, line); // Throw away header;

	// Read in each line of the file
	while (getline(input_file, line))
	{

		// Extract trajectory information from line.
		// Test input is of the form:
		// trajectory_number, x, y, t

		point point; // Variable to read line information into

		// Parse each input line into point
		char * cstr = new char [line.length()+1]; // Convert to line for tokenization. Not the best way to do this I know.
		std::strcpy(cstr, line.c_str());
		point.trajectory_number = std::stoi(strtok(cstr, ","));
		point.x = std::stod(strtok(NULL, ","));
		point.y = std::stod(strtok(NULL, ","));
		point.t = std::stod(strtok(NULL, ","));

		delete[] cstr;

		// Add point to database
		host_trajectory_data.push_back(point);
	}

	input_file.close();

	if(DEBUG)
	{
		std::cout << "Finished reading in a file with contents: " << std::endl;
		// Did the file read correctly?
		for (int i = 0; i < host_trajectory_data.size(); i++) {
			point point = host_trajectory_data.at(i);
			std::cout << "Traj_num: " << point.trajectory_number << " x: " << point.x << " y: " << point.y << " t: " << point.t << std::endl;
		}
	}



	return 0;
}
