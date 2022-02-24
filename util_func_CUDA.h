// These functions should run on GPU. 

#pragma once
#ifndef CUDA_FUNC_DEFINE_H
#define CUDA_FUNC_DEFINE_H

#include"util_class.h"
#include<stdio.h>
#include<algorithm>
#include<cmath>
#include<cuda_runtime.h>
#include"device_launch_parameters.h"
#include<thrust/tuple.h>

#define Pi 3.14159265  // PI
#define eps 1e-8  // Tolerance
#define ADD_FACE_MAX_NUM 40  // Maximum number of neighboring faces considered
#define ADD_C_MAX_NUM 15  // Maximum number of neighboring vertices considered
#define MID_NUM 4  // Number of points considered on bisector
#define default_elem_size 0.05  // Default uniform mesh size
#define inter_max_dist_square 25 * 0.05 * 0.05  // (5 * 0.05) ^ 2

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

// using namespace std;

// User-defined element size distribution, GPU version
__device__ __forceinline__ float cuda_elem_size(float get_x, float get_y)
{
	float final_elem_size = default_elem_size;
	return final_elem_size;
}

// Element quality measure on GPU
// See section 2.2.2, "Quality measure" part for more details. 
__device__ __forceinline__ float cuda_elem_quality(float rho1, float rho2, float smallest_angle,
	float closest_dist, float largest_angle, vertex v0 = vertex(0, 0), vertex v1 = vertex(0.6, 0),
	vertex v2 = vertex(0, 0.6))
{
	// Three vertices
	float A_x = v0.x;
	float A_y = v0.y;
	float B_x = v1.x;
	float B_y = v1.y;
	float C_x = v2.x;
	float C_y = v2.y;

	float l_AC = sqrt(pow(A_x - C_x, 2) + pow(A_y - C_y, 2));
	float delta1 = min(l_AC / rho1, rho1 / l_AC);
	float l_BC = sqrt(pow(B_x - C_x, 2) + pow(B_y - C_y, 2));
	float delta2 = min(l_BC / rho2, rho2 / l_BC);

	float l_AB = sqrt(pow(A_x - B_x, 2) + pow(A_y - B_y, 2));
	float AB_cross_AC = (B_x - A_x) * (C_y - A_y) - (B_y - A_y) * (C_x - A_x);
	float alpha = abs(AB_cross_AC) / (l_AB * l_AB + l_BC * l_BC + l_AC * l_AC);

	// Smallest angle penalty
	float angle_penalty = 1.0;
	if (smallest_angle < 60.0 && smallest_angle > 0.0)
	{
		angle_penalty += 2.0 * (60.0 - smallest_angle) / 60.0;  // 1.0 ~ 3.0
	}

	// Largest angle penalty
	float large_angle_penalty = 1.0;
	if (largest_angle > 90.0)
	{
		large_angle_penalty += 2.0 * (largest_angle - 90.0) / 90.0;  // 1.0 ~ 3.0
	}

	// Closest distance penalty
	float dist_penalty = 1.0;
	float dist_penalty_threshold = 0.55 * rho1;
	if (closest_dist < dist_penalty_threshold && closest_dist > 0.0)
	{
		dist_penalty += 7.0 * (dist_penalty_threshold - closest_dist) / dist_penalty_threshold;
	}

	return (((alpha * delta1 * delta2) / angle_penalty) / large_angle_penalty) / dist_penalty;
}

// Calculate expected height of the new element
__device__ __forceinline__ float cuda_triangle_height(float l_ab, float local_size)
{
	float new_waist = 0.0;  // First calculate expected waist length
	float waist_factor = 1 / 0.55;  // see section 2.2.2 for details
	if ((1 / waist_factor) * l_ab >= local_size)
		new_waist = l_ab / waist_factor;
	else if (waist_factor * l_ab < local_size)
		new_waist = waist_factor * l_ab;
	else
		new_waist = local_size;
	float new_height = sqrt(new_waist * new_waist - 0.25 * l_ab * l_ab);
	return new_height;
}

// Distance between two vertices
__device__ __forceinline__ float cuda_distance_vert_vert(vertex va, vertex vb)
{
	return sqrt(pow(vb.x - va.x, 2) + pow(vb.y - va.y, 2));
}

// Distance between vertex and face
__device__ float cuda_distance_vert_face(vertex v0, face tmp_face)
{
	float A2 = tmp_face.Area * tmp_face.Area;
	float t = max(float(0.0), min(float(1.0), ((v0.x - tmp_face.s.x) * tmp_face.es_x +
		(v0.y - tmp_face.s.y) * tmp_face.es_y) / A2));
	// First project v0 onto face
	float v0_proj_x = tmp_face.s.x + t * tmp_face.es_x;
	float v0_proj_y = tmp_face.s.y + t * tmp_face.es_y;
	return sqrt(pow(v0.x - v0_proj_x, 2) + pow(v0.y - v0_proj_y, 2));
}

// Find faces that are close to a given center vertex
// Consider a circular region, centered at "tmp_center", with radius "radius". 
__device__ __forceinline__ int cuda_get_faces_in_radius(int id_x, vertex tmp_center,
	float radius, QuadTreeNode* front_tree, face* add_face_arr, int* add_id_arr)
{
	myRectangle range = myRectangle(tmp_center.x - radius,
		tmp_center.y - radius, 2 * radius, 2 * radius);

	int add_num = 0;  // Total number of faces that satisfy our requirement
	int s[50] = {};  // Use s[] to simulate a stack
	int s_len = 1;
	s[0] = 0;

	while (s_len > 0)
	{
		if (s_len > 50)  // Length exceeds limit
		{
			printf("stack length exceeds limit!\n");
		}

		int top = s[s_len - 1];
		QuadTreeNode check_node = front_tree[top];

		if (check_node.face_num == 1)
		{
			face p = check_node.my_face;

			if (range.contains(p.mid_v))
			{
				add_face_arr[add_num] = p;
				add_id_arr[add_num] = top;
				add_num += 1;
			}
		}
		s_len -= 1;  // Pop out of stack s[]

		if (check_node.left_child >= 0)  // Iterate over all children
		{
			if (front_tree[check_node.left_child].boundary.intersects(range))
			{
				// Start from left-child
				s[s_len] = check_node.left_child;
				s_len += 1;  // Push into stack s[]
			}

			int node_id = check_node.left_child;

			for (int i = 0; i < 3; i++)  // Go over other children sequentially
			{
				node_id++;  // Move to right sibling
				if (front_tree[node_id].boundary.intersects(range))
				{
					s[s_len] = node_id;
					s_len += 1;
				}
			}
		}
	}
	return add_num;
}

// Judge whether two vertices are equivalent, GPU version. 
__device__ __forceinline__ bool cuda_vert_equal(vertex va, vertex vb)
{
	if (abs(va.x - vb.x) < eps && abs(va.y - vb.y) < eps)
		return true;
	return false;
}

// Judge whether two faces are equivalent, GPU version. 
__device__ __forceinline__ bool cuda_face_equal(face p, face q)
{
	if (cuda_vert_equal(p.s, q.s) && cuda_vert_equal(p.e, q.e))
		return true;
	else if (cuda_vert_equal(p.s, q.e) && cuda_vert_equal(p.e, q.s))
		return true;
	else
		return false;
}

// Check which side of face s is vertex v in (a face/line divides the domain into 2 parts). 
// Return "true" when the normal vector points to v. 
__device__ __forceinline__ bool cuda_same_dir(face s, vertex v)
{
	if ((v.x - s.mid_x) * s.N_x + (v.y - s.mid_y) * s.N_y < eps)
	{
		return false;
	}
	else
	{
		return true;
	}
}

// Look for new vertices (denoted as "c" here) on existing faces
// See section 2.2.2, "New vertices on existing faces" for more details. 
__device__ __forceinline__ int cuda_get_potent_c(int id_x, face* add_face_arr,
	int face_num, face grow_face, vertex tmp_center, float radius, vertex* front_c_arr)
{
	int add_c_num = 0;  // Total number of vertices that saatisfy our requirement.
	for (int i = 0; i < face_num; i++)  // Go over all the neighboring faces
	{
		face tmp_face = add_face_arr[i];  // check tmp_face.s
		if (cuda_distance_vert_vert(tmp_face.s, tmp_center) < radius)
		{
			if (cuda_vert_equal(tmp_face.s, grow_face.s) ||
				cuda_vert_equal(tmp_face.s, grow_face.e))
			{
				;
			}
			else if (cuda_same_dir(grow_face, tmp_face.s))
			{
				// Satisfy requirements, save the vertex.
				front_c_arr[add_c_num] = tmp_face.s;
				front_c_arr[add_c_num].if_mid = false;
				add_c_num += 1;
			}
		}
	}
	return add_c_num;
}

// Look for new vertices (denoted as "i" here) on perpendicular bi-sector
// See section 2.2.2, "New vertices on perpendicular bisector" for more details. 
__device__ __forceinline__ void cuda_get_potent_i(int id_x, int N_mid, face grow_face,
	vertex tmp_center, vertex* front_i_arr)
{
	for (int mid_i = 0; mid_i < N_mid; mid_i++)
	{
		float ratio = 1.0 - pow(0.85, mid_i);
		// Consider: ratio * tmp_edge.mid_v + (1 - ratio) * tmp_center
		float potent_i_x = ratio * grow_face.mid_x + (1 - ratio) * tmp_center.x;
		float potent_i_y = ratio * grow_face.mid_y + (1 - ratio) * tmp_center.y;
		vertex potent_i = vertex(potent_i_x, potent_i_y);
		potent_i.if_mid = true;  // This vertex lies on bisector
		front_i_arr[mid_i] = potent_i;
	}
}

// Judge whether two faces intersect, GPU version. 
__device__ __forceinline__ bool cuda_face_intersect(face p, face q)
{
	// Two line segments
	double crit1_val1 = p.es_x * (q.s.y - p.s.y) - (q.s.x - p.s.x) * p.es_y;
	double crit1_val2 = p.es_x * (q.e.y - p.s.y) - (q.e.x - p.s.x) * p.es_y;

	bool crit1 = (crit1_val1 * crit1_val2 < -1e-4 * eps);
	if (!crit1)
		return false;

	double crit2_val1 = q.es_x * (p.s.y - q.s.y) - (p.s.x - q.s.x) * q.es_y;
	double crit2_val2 = q.es_x * (p.e.y - q.s.y) - (p.e.x - q.s.x) * q.es_y;

	bool crit2 = (crit2_val1 * crit2_val2 < -1e-4 * eps);
	if (!crit2)
		return false;

	return true;  // Intersect with each other
}

// Check whether the 2 new faces (s1 and s2) intersect with any face in add_face_arr.  
__device__ __forceinline__ bool cuda_no_intersect(int id_x, face s1, face s2, face grow_face,
	face* add_face_arr, int add_num)
{
	for (int i = 0; i < add_num; i++)  // Iterate over all saved faces
	{
		face tmp_face = add_face_arr[i];

		if (cuda_face_intersect(s1, tmp_face) || cuda_face_intersect(s2, tmp_face))
		{
			return false;  // This new element is illegal
		}
	}
	return true;  // No intersection: this new element remains legal
}

// Calculate the angle between two faces, in degrees. 
__device__ __forceinline__ float cuda_face_angle(face p, face q)
{
	// Calculate the inner product first
	float tmp_product = (p.es_x * q.es_x + p.es_y * q.es_y) / (p.Area * q.Area);  // Normalize: cos

	if (cuda_vert_equal(p.s, q.e) || cuda_vert_equal(p.e, q.s))
		tmp_product *= -1;
	// Clip to [-1, 1]
	if (tmp_product < -1.0)
		tmp_product = -1.0;
	if (tmp_product > 1.0)
		tmp_product = 1.0;

	float angle = acos(tmp_product) * (180.0 / Pi);  // Transform: [0, pi] -> [0, 180]
	return angle;
}

// Check: the angle between s1 (or s2) and grow_face should be smaller than angle between
// some neighboring face and grow_face. 
// For more details, see section 2.2.2, the second condition. 
__device__ __forceinline__ thrust::tuple<bool, float, float> cuda_no_out_range(int id_x,
	face s1, face s2, face grow_face, face* add_face_arr, int add_num)
{
	bool no_out_angle = true;

	float new_theta_1 = cuda_face_angle(s1, grow_face);  // s1 -> theta_1
	float new_theta_2 = cuda_face_angle(s2, grow_face);  // s2 -> theta_2

	float smallest_angle = 360;  // Record smallest angle between new faces and existing faces.
	// largest angle: max(theta_1, theta_2)
	float largest_angle = new_theta_1 + (new_theta_2 - new_theta_1) * int(new_theta_2 > new_theta_1);

	for (int i = 0; i < add_num; i++)  // Iterate over all saved faces. 
	{
		face tmp_face = add_face_arr[i];

		if (cuda_face_equal(tmp_face, grow_face))  // Skip the face you grow from. 
			continue;

		// Face 1: connected with grow_face.s
		if (cuda_vert_equal(tmp_face.s, grow_face.s) ||
			cuda_vert_equal(tmp_face.e, grow_face.s))
		{
			bool ss_equal = cuda_vert_equal(tmp_face.s, grow_face.s);

			float check_same_side_1 = (tmp_face.s.x + tmp_face.es_x * int(ss_equal) - grow_face.mid_x)
				* grow_face.N_x + (tmp_face.s.y + tmp_face.es_y * int(ss_equal) - grow_face.mid_y)
				* grow_face.N_y;

			float angle_1 = cuda_face_angle(tmp_face, grow_face);
			if (angle_1 < new_theta_1 - 1e-4)  // angle_1 < theta_1, illegal.
			{
				if (check_same_side_1 > eps)
				{
					no_out_angle = false;
					break;
				}
			}
			else
			{
				if (angle_1 - new_theta_1 > 1e-4 && angle_1 - new_theta_1 < smallest_angle)
				{
					smallest_angle = angle_1 - new_theta_1;  // Update smallest angle
				}
			}
		}

		// Face 2: connected with grow_face.e
		if (cuda_vert_equal(tmp_face.s, grow_face.e) ||
			cuda_vert_equal(tmp_face.e, grow_face.e))
		{
			bool se_equal = cuda_vert_equal(tmp_face.s, grow_face.e);
			float check_same_side_2 = (tmp_face.s.x + tmp_face.es_x * int(se_equal) - grow_face.mid_x) * grow_face.N_x
				+ (tmp_face.s.y + tmp_face.es_y * int(se_equal) - grow_face.mid_y) * grow_face.N_y;

			float angle_2 = cuda_face_angle(tmp_face, grow_face);
			if (angle_2 < new_theta_2 - 1e-4)  // angle_2 < theta_2, illegal.
			{
				if (check_same_side_2 > eps)
				{
					no_out_angle = false;
					break;
				}
			}
			else
			{
				if (angle_2 - new_theta_2 > 1e-4 && angle_2 - new_theta_2 < smallest_angle)
				{
					smallest_angle = angle_2 - new_theta_2;  // Update smallest angle
				}
			}
		}
	}
	return thrust::make_tuple(no_out_angle, smallest_angle, largest_angle);
}

// Judge whether two faces share a same node
// If these two faces are the same face, return false!
__device__ __forceinline__ bool cuda_face_same_node(face p, face q)
{
	// Based on distance between vertices
	if (cuda_distance_vert_vert(p.s, q.s) < eps)
	{
		if (cuda_distance_vert_vert(p.e, q.e) > eps)
			return true;
		else
			return false;
	}
	if (cuda_distance_vert_vert(p.s, q.e) < eps)
	{
		if (cuda_distance_vert_vert(p.e, q.s) > eps)
			return true;
		else
			return false;
	}
	return false;
}

// Check whether s1 (or s2) forms very small angle with some existing face. 
// If very small angle is formed: the new element is illegal. 
__device__ __forceinline__ bool cuda_no_small_angle(int id_x, face s1, face s2,
	face* add_face_arr, int add_num)
{
	bool no_small_angle_bool = true;

	float min_angle = 6;  // threshold set as 6 degrees
	for (int i = 0; i < add_num; i++)  // Iterate over all saved faces. 
	{
		face tmp_face = add_face_arr[i];

		if (cuda_face_same_node(s1, tmp_face))  // Check s1
		{
			float angle_1 = cuda_face_angle(s1, tmp_face);
			if (0.001 < angle_1 && angle_1 < min_angle)
			{
				no_small_angle_bool = false;
				break;
			}
		}

		if (cuda_face_same_node(s2, tmp_face))  // Check s2
		{
			float angle_2 = cuda_face_angle(s2, tmp_face);
			if (0.001 < angle_2 && angle_2 < min_angle)
			{
				no_small_angle_bool = false;
				break;
			}
		}
	}
	return no_small_angle_bool;
}

// Check whether some existing vertex is too close to generated new faces. 
__device__ __forceinline__ thrust::tuple<bool, float> cuda_no_too_close(int id_x, face new_face,
	face* add_face_arr, int add_num, float dist_threshold)
{
	bool no_too_close_bool = true;
	float min_dist = 100.0;  // Record the minimum distance

	for (int i = 0; i < add_num; i++)
	{
		face tmp_face = add_face_arr[i];  // Check vertex tmp_face.s
		if (cuda_vert_equal(tmp_face.s, new_face.s) ||
			cuda_vert_equal(tmp_face.s, new_face.e))
			continue;

		// If some vertex is too close to formed face: reject
		// No need to consider initial front faces
		float dist = cuda_distance_vert_face(tmp_face.s, new_face);
		if (dist < dist_threshold)
		{
			no_too_close_bool = false;
			break;
		}
		min_dist -= (min_dist - dist) * int(dist < min_dist);  // Update closest_dist
	}
	return thrust::make_tuple(no_too_close_bool, min_dist);
}

// Check whether vertex tmp_i is too close to some existing face. 
__device__ __forceinline__ bool cuda_no_too_close_face(int id_x, vertex tmp_i, face* add_face_arr,
	int add_num, float min_dist)
{
	for (int i = 0; i < add_num; i++)  // Iterate over all saved faces. 
	{
		face tmp_face = add_face_arr[i];

		// If some vertex is too close to existing face: reject
		float dist_i = cuda_distance_vert_face(tmp_i, tmp_face);
		if (dist_i < min_dist)
			return false;
	}
	return true;
}

// Find the best candidate location on existing front.
__device__ __forceinline__ thrust::tuple<vertex, float, bool> cuda_find_good_c(int id_x,
	vertex* front_c_arr,
	int add_c_num,
	face grow_face,
	float elem_size,
	face* add_face_arr,
	int add_num)
{
	float optim_c_quality = 0.0;  // Record best quality
	vertex optim_c = vertex(0, 0);  // Record the legal vertex that leads to best quality
	bool c_exist = false;  // Record whether we've found a candidate

	vertex tmp_c;
	face s1, s2;

	for (int i = 0; i < add_c_num; i++)
	{
		tmp_c = front_c_arr[i];

		s1 = face(grow_face.s, tmp_c);
		s2 = face(tmp_c, grow_face.e);

		// Check intersection

		if (!cuda_no_intersect(id_x, s1, s2, grow_face, add_face_arr, add_num))
		{
			continue;  // Next c candidate
		}

		auto no_out_range_result = cuda_no_out_range(id_x, s1, s2, grow_face, add_face_arr, 
			add_num);
		bool no_out_range_bool = thrust::get<0>(no_out_range_result);
		float smallest_angle = thrust::get<1>(no_out_range_result);
		float largest_angle = thrust::get<2>(no_out_range_result);

		if (!no_out_range_bool)
		{
			continue;  // Next c candidate
		}

		// Check small angle
		if (!cuda_no_small_angle(id_x, s1, s2, add_face_arr, add_num))
		{
			continue;  // Next c candidate
		}

		// Check distance: other verts to face 1 / face 2
		auto no_too_close_result_1 = cuda_no_too_close(id_x, s1, add_face_arr, add_num, 
			0.09 * elem_size);
		bool no_too_close_bool_1 = thrust::get<0>(no_too_close_result_1);
		float closest_dist_1 = thrust::get<1>(no_too_close_result_1);
		if (!no_too_close_bool_1)
		{
			continue;  // Next c candidate
		}

		auto no_too_close_result_2 = cuda_no_too_close(id_x, s2, add_face_arr, add_num, 
			0.09 * elem_size);
		bool no_too_close_bool_2 = thrust::get<0>(no_too_close_result_2);
		float closest_dist_2 = thrust::get<1>(no_too_close_result_2);
		if (!no_too_close_bool_2)
		{
			continue;  // Next c candidate
		}

		float closest_dist = closest_dist_1 - (closest_dist_1 - closest_dist_2) * 
			int(closest_dist_1 > closest_dist_2);

		// Get here: good candidate!
		c_exist = true;
		float tmp_quality = cuda_elem_quality(elem_size, elem_size, smallest_angle, closest_dist,
			largest_angle, grow_face.s, grow_face.e, tmp_c);  // Calculate the quality measure
		if (tmp_quality > optim_c_quality)  // Only keep the element with best quality
		{
			optim_c_quality = tmp_quality;
			optim_c = tmp_c;
		}
	}

	return thrust::make_tuple(optim_c, optim_c_quality, c_exist);
}

// Find the best candidate location on perpendicular bisector. 
__device__ __forceinline__ thrust::tuple<vertex, float, bool> cuda_find_good_i(int id_x,
	int L_mid,
	vertex* front_i_arr,
	face grow_face,
	float elem_size,
	face* add_face_arr,
	int add_num)
{
	float optim_i_quality = 0.0;  // Record best quality
	vertex optim_i = vertex(0, 0);  // Record the legal vertex that leads to best quality
	bool i_exist = false;  // Record whether we've found a candidate

	for (int i = 0; i < L_mid; i++)
	{
		vertex tmp_i = front_i_arr[i];
		face s1 = face(grow_face.s, tmp_i);
		face s2 = face(tmp_i, grow_face.e);

		// Check intersection
		if (!cuda_no_intersect(id_x, s1, s2, grow_face, add_face_arr, add_num))
		{
			continue;  // Next i candidate
		}

		auto no_out_range_result = cuda_no_out_range(id_x, s1, s2, grow_face, add_face_arr, 
			add_num);
		bool no_out_range_bool = thrust::get<0>(no_out_range_result);
		float smallest_angle = thrust::get<1>(no_out_range_result);
		float largest_angle = thrust::get<2>(no_out_range_result);
		if (!no_out_range_bool)
		{
			continue;  // Next i candidate
		}

		// Check distance: i to other vertices
		if (!cuda_no_too_close_face(id_x, tmp_i, add_face_arr, add_num,
			0.4 * elem_size))
		{
			continue;  // Next i candidate
		}

		// Check distance: other vertices to face s1 / s2
		auto no_too_close_result = cuda_no_too_close(id_x, s1, add_face_arr, add_num, 
			0.35 * s1.Area);
		bool no_too_close_bool = thrust::get<0>(no_too_close_result);
		float closest_dist_1 = thrust::get<1>(no_too_close_result);
		if (!no_too_close_bool)
		{
			continue;  // Next i candidate
		}

		no_too_close_result = cuda_no_too_close(id_x, s2, add_face_arr, add_num, 0.35 * s2.Area);  // 0.25
		no_too_close_bool = thrust::get<0>(no_too_close_result);
		float closest_dist_2 = thrust::get<1>(no_too_close_result);
		if (!no_too_close_bool)
		{
			continue;  // Next i candidate
		}

		float closest_dist = closest_dist_1 + (closest_dist_2 - closest_dist_1) * 
			int(closest_dist_1 > closest_dist_2);

		// Get here: good candidate!
		i_exist = true;
		float tmp_quality = cuda_elem_quality(elem_size, elem_size, smallest_angle, closest_dist,
			largest_angle, grow_face.s, grow_face.e, tmp_i);  // Calculate the quality measure
		if (tmp_quality > optim_i_quality)  // Only keep the element with best quality
		{
			optim_i_quality = tmp_quality;
			optim_i = tmp_i;
		}
	}

	return thrust::make_tuple(optim_i, optim_i_quality, i_exist);
}

// After all c's (on existing faces) and i's (on the perpendicular bisector) have been collected,
// we need to find the best candidate based on quality measure. 
__device__ __forceinline__ thrust::tuple <vertex, bool, bool> cuda_choose_c_i(bool c_exist, 
	bool i_exist, float optim_c_quality, float optim_i_quality, vertex optim_c, vertex optim_i)
{
	vertex new_vertex = vertex(0.0, 0.0);  // Record the best candidate
	bool find_new_vertex = false;
	bool is_existing_vertex = false;  // Record whether this candidate is an existing vertex. 

	if (c_exist || i_exist)
		find_new_vertex = true;

	if (c_exist)
	{
		if (i_exist)
		{
			if (optim_i_quality > optim_c_quality)  // Choose i
			{
				new_vertex = optim_i;
			}
			else  // Choose c
			{
				new_vertex = optim_c;
				is_existing_vertex = true;
			}
		}
		else  // no i, choose c
		{
			new_vertex = optim_c;
			is_existing_vertex = true;
		}
	}
	else
	{
		if (i_exist)  // no c, choose i
		{
			new_vertex = optim_i;
		}
	}

	return thrust::make_tuple(new_vertex, find_new_vertex, is_existing_vertex);
}

// Check whether new face s already exists in add_face_arr. 
// Return both boolean answer and ID (set as -1 by default). 
__device__ __forceinline__ thrust::tuple<bool, int> cuda_if_existing_face(int id_x, face s,
	face* add_face_arr, int* add_id_arr, int add_num)
{
	bool existing_face_bool = false;
	int	existing_face_id = -1;  // Record the searched ID

	for (int i = 0; i < add_num; i++)
	{
		face tmp_face = add_face_arr[i];

		if (abs(tmp_face.Area - s.Area) > 1e-4)  // Check face area (length) first
		{
			continue;
		}
		else if (cuda_face_equal(tmp_face, s))
		{
			existing_face_bool = true;
			existing_face_id = add_id_arr[i];
			break;  // Found face s inside add_face_arr!
		}
	}
	return thrust::make_tuple(existing_face_bool, existing_face_id);
}

#endif
