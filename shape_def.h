#pragma once
#ifndef SHAPE_DEFINE_H
#define SHAPE_DEFINE_H

#include<iostream>
#include<vector>
#include"util_class.h"
#include"util_func.h"

#define Pi 3.14159265  // PI

// Double ellipses
// Two ellipses, both centered at the origin. 
// Size given by (a_in, b_in) and (a_out, b_out)
// Mesh the region in between
std::vector<face> double_ellipses(float a_in, float b_in, float a_out, float b_out)
{
	// Define initial boundaries

	// First handle the outer ellipse
	float round_len_out = Pi * (3 * (a_out + b_out) -
		sqrt((3 * a_out + b_out) * (a_out + 3 * b_out)));  // Ellipse's perimeter (approximately)
	int N_out = floor(round_len_out / elem_size(b_out, 0.0));  // Number of initial faces

	std::vector<float> theta_out(N_out);
	std::vector<float> cir_x_out(N_out);
	std::vector<float> cir_y_out(N_out);
	// Generate N_out vertices on the outer ellipse
	for (int i = 0; i < N_out; i++)
	{
		theta_out[i] = (float(i) / N_out) * 2 * Pi;
		cir_x_out[i] = a_out * cos(theta_out[i]);
		cir_y_out[i] = b_out * sin(theta_out[i]);
	}

	// Next handle the inner ellipse
	float round_len_in = Pi * (3 * (a_in + b_in) - 
		sqrt((3 * a_in + b_in) * (a_in + 3 * b_in)));  // Ellipse's perimeter (approximately)
	int N_in = floor(round_len_in / elem_size(b_in, 0.0));  // Number of initial faces

	std::vector<float> theta_in(N_in);
	std::vector<float> cir_x_in(N_in);
	std::vector<float> cir_y_in(N_in);
	// Generate N_in vertices on the inner ellipse
	for (int i = 0; i < N_in; i++)
	{
		theta_in[i] = (float(i) / N_in) * 2 * Pi;
		cir_x_in[i] = a_in * cos(theta_in[i]);
		cir_y_in[i] = b_in * sin(theta_in[i]);
	}

	std::vector<face> init_face_list(N_in + N_out);

	for (int i = 0; i < N_out; i++)  // Outer ellipse, normal vectors pointing inward
	{
		vertex tmp_s = vertex(cir_x_out[i], cir_y_out[i], i);
		vertex tmp_e = vertex(cir_x_out[(i + 1) % N_out], cir_y_out[(i + 1) % N_out], 
			(i + 1) % N_out);
		face tmp_bound = face(tmp_s, tmp_e);
		init_face_list[i] = tmp_bound;
	}

	for (int i = 0; i < N_in; i++)  // Inner ellipse, normal vectors pointing outward
	{
		vertex tmp_s = vertex(cir_x_in[(i + 1) % N_in], cir_y_in[(i + 1) % N_in], N_out + i);
		vertex tmp_e = vertex(cir_x_in[i], cir_y_in[i], N_out + (i + N_in - 1) % N_in);
		face tmp_bound = face(tmp_s, tmp_e);
		init_face_list[i + N_out] = tmp_bound;
	}

	return init_face_list;
}

#endif