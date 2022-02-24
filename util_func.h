// These functions should run on CPU. 

#pragma once
#ifndef FUNC_DEFINE_H
#define FUNC_DEFINE_H

#define eps 1e-8  // Tolerance
#define default_elem_size 0.05  // Default element size for "ring" is 0.05

#include<graphics.h>
#include "util_class.h"

// Plot the current front
// I'm using EasyX library for plotting figures. 
// You can see the docs at https://docs.easyx.cn/zh-cn
// Other libraries will also work, since plotting (usually just for debugging / visualization)
// does not affect SAFT algorithm.
void visualize_front(QuadTree front_tree, std::vector<face> archived_face_list,
	std::vector<int> grow_id_arr, float pause_time, float xs, float xe,
	float ys, float ye)
{
	// Size of the region to be visualized
	float actual_w = xe - xs;
	float actual_h = ye - ys;
	// Center of the region to be visualized
	float mid_x = 0.5 * (xe + xs);
	float mid_y = 0.5 * (ys + ye);

	// Resolution: number of pixels
	int graph_w = 1200;
	int graph_h = int(graph_w * actual_h / actual_w);

	float dx = actual_w / graph_w;
	float dy = dx;

	initgraph(graph_w, graph_h);

	int num_node = front_tree.node_num;
	int num_archived = archived_face_list.size();  // Number of archived faces

	// Plot the archived faces
	for (int i = 0; i < num_archived; i++)
	{
		face plot_face = archived_face_list[i];
		int sx = int((graph_w / 2) + (plot_face.s.x - mid_x) / dx);
		int sy = int((graph_h / 2) - (plot_face.s.y - mid_y) / dy);
		int ex = int((graph_w / 2) + (plot_face.e.x - mid_x) / dx);
		int ey = int((graph_h / 2) - (plot_face.e.y - mid_y) / dy);

		if ((sx > 0 && sx < graph_w) && (ex > 0 && ex < graph_w))
		{
			if ((sy > 0 && sy < graph_h) && (ey > 0 && ey < graph_h))
			{
				setlinecolor(RGB(30, 255, 50));  // Archive faces marked with green
				line(sx, sy, ex, ey);
			}
		}
	}

	// Plot the active faces
	for (int i = 0; i < num_node; i++)  // Go over the QuadTree
	{
		QuadTreeNode tmp_node = front_tree.node_arr[i];
		if (tmp_node.face_num == 1)  // This node contains face
		{
			face plot_face = tmp_node.my_face;

			int sx = int((graph_w / 2) + (plot_face.s.x - mid_x) / dx);
			int sy = int((graph_h / 2) - (plot_face.s.y - mid_y) / dy);
			int ex = int((graph_w / 2) + (plot_face.e.x - mid_x) / dx);
			int ey = int((graph_h / 2) - (plot_face.e.y - mid_y) / dy);

			if ((sx > 0 && sx < graph_w) && (ex > 0 && ex < graph_w))
			{
				if ((sy > 0 && sy < graph_h) && (ey > 0 && ey < graph_h))
				{
					setlinecolor(RGB(50, 100, 255));  // Active faces marked with blue
					line(sx, sy, ex, ey);
				}
			}
		}
	}

	// Plot the P faces randomly chosen
	int len_grow_id_arr = grow_id_arr.size();
	for (int i = 0; i < len_grow_id_arr; i++)
	{
		int tmp_id = grow_id_arr[i];
		if (tmp_id >= front_tree.node_num)
			continue;
		QuadTreeNode tmp_node = front_tree.node_arr[tmp_id];
		if (tmp_node.face_num == 1)
		{
			face plot_face = tmp_node.my_face;
			int sx = int((graph_w / 2) + (plot_face.s.x - mid_x) / dx);
			int sy = int((graph_h / 2) - (plot_face.s.y - mid_y) / dy);
			int ex = int((graph_w / 2) + (plot_face.e.x - mid_x) / dx);
			int ey = int((graph_h / 2) - (plot_face.e.y - mid_y) / dy);

			if ((sx > 0 && sx < graph_w) && (ex > 0 && ex < graph_w))
			{
				if ((sy > 0 && sy < graph_h) && (ey > 0 && ey < graph_h))
				{
					setlinecolor(RGB(255, 0, 0));  // Chosen faces marked with red
					line(sx, sy, ex, ey);
				}
			}
		}
	}

	// Press any key to continue
	getchar();  // prevent from exiting automatically

	return;
}

// User-defined element size distribution, CPU version
float elem_size(float get_x, float get_y)  // return the element size around (get_x, get_y)
{
	// You can tune this to be any arbitrary user-defined size distributiion. 
	float local_size = default_elem_size;  // This leads to a uniform mesh
	return local_size;
}

// Check whether two vertices are equal
bool vertex_equal(vertex va, vertex vb)
{
	if (std::abs(va.x - vb.x) < eps && std::abs(va.y - vb.y) < eps)
		return true;
	return false;
}

// Check whether two faces are equal
// by checking the coordinates of vertices
bool face_equal(face p, face q)
{
	// Check whether two vertices coincide
	if (vertex_equal(p.s, q.s) && vertex_equal(p.e, q.e))  // Same direction
		return true;
	else if (vertex_equal(p.s, q.e) && vertex_equal(p.e, q.s))  // Different direction
		return true;
	else
		return false;
}

// Check which side of face s is vertex v in.
// Return "true" when the normal vector points to v. 
bool same_dir(face s, vertex v)
{
	// Check the inner product
	if ((v.x - s.mid_x) * s.N_x + (v.y - s.mid_y) * s.N_y <= 0.0)
		return false;
	return true;
}

// Randomly sample k active faces from front_tree. 
// Return a vector<int>, containing IDs of all sampled nodes/faces. 
std::vector<int> fast_sample(QuadTree front_tree, int k)
{
	int node_num = front_tree.node_num;
	std::vector<int> available_id_vector;  // Check which nodes contain active faces

	for (int i = 0; i < node_num; i++)  // Iterate over all QuadTreeNodes
	{
		if (front_tree.node_arr[i].face_num == 1)
			available_id_vector.push_back(i);
	}

	int n = available_id_vector.size(); // Choose k from n
	if (k >= n)
		return available_id_vector;
	else  // k < n
	{
		int l = 0;  // Record the length of result_arr. 
		std::vector<bool> choose_bool_arr(n);
		std::vector<int> result_arr;  // Record the chosen IDs. 
		while (l < k)
		{
			int i = rand() % n;  // random number: 0,1,...,n-1
			if (!choose_bool_arr[i])  // Hasn't been chosen yet: then choose it. 
			{
				result_arr.push_back(available_id_vector[i]);
				choose_bool_arr[i] = true;
				l += 1;
			}
		}
		return result_arr;
	}
}

#endif
