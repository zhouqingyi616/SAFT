#pragma once
#ifndef CLASS_DEFINE_H
#define CLASS_DEFINE_H

#include<vector>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#ifndef max
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

class vertex
{
public:
	// Coordinates
	float x;
	float y;
	int id;  // ID of vertex
	// Whether this new vertex lies on bisector of face. 
	// See section 2.2.2 in the paper ("New vertices on perpendicular bisector")
	bool if_mid;

	CUDA_CALLABLE_MEMBER vertex()
	{
		x = 0.0;
		y = 1.0;
		id = -1;
		if_mid = true;
	}

	CUDA_CALLABLE_MEMBER ~vertex() {}

	CUDA_CALLABLE_MEMBER vertex(float x0, float y0, int id0 = -1)  // Constructor
	{
		x = x0;
		y = y0;
		id = id0;
		if_mid = true;
	}

	CUDA_CALLABLE_MEMBER void operator=(const vertex& v)
	{
		x = v.x;
		y = v.y;
		id = v.id;
		if_mid = v.if_mid;
	}
};

class id_pair  // pair of integers, used to record faces
{
public:
	int s_id;  // ID of starting vertex
	int e_id;  // ID of ending vertex

	CUDA_CALLABLE_MEMBER id_pair(int s_id0, int e_id0)
	{
		s_id = s_id0;
		e_id = e_id0;
	}

	CUDA_CALLABLE_MEMBER ~id_pair() {}
};

class face
{
public:
	vertex s;  // Starting vertex
	vertex e;  // Ending vertex
	float Area;  // Area (actually length in 2D)
	float N_x, N_y;  // Normal vector

	vertex mid_v;  // Mid point
	float mid_x, mid_y;  // Coordinates of mid point

	float es_x, es_y;  // 2D vector: end - start

	// Bounding box
	float bb_xs, bb_xe;  // starting x, ending x
	float bb_ys, bb_ye;  // starting y, ending y

	// Whether this face already exists
	int existed_id;  // if not (by default), then existed_id = -1

	CUDA_CALLABLE_MEMBER face(vertex s0 = vertex(0, 0), vertex e0 = vertex(0, 1),
		int id0 = -1)
	{
		s = s0;
		e = e0;

		// Mid point
		mid_x = (s0.x + e0.x) / 2;
		mid_y = (s0.y + e0.y) / 2;
		mid_v = vertex(mid_x, mid_y);

		// Bounding box
		bb_xs = min(s0.x, e0.x);
		bb_xe = max(s0.x, e0.x);

		bb_ys = min(s0.y, e0.y);
		bb_ye = max(s0.y, e0.y);

		// e_s vector
		es_x = e0.x - s0.x;
		es_y = e0.y - s0.y;


		// Area
		Area = sqrt(es_x * es_x + es_y * es_y);

		// Norm vector
		N_x = -(e0.y - s0.y) / Area;
		N_y = (e0.x - s0.x) / Area;

		// Existed id
		existed_id = id0;
	}

	CUDA_CALLABLE_MEMBER ~face() {}

	CUDA_CALLABLE_MEMBER void operator=(const face& f)
	{
		s = f.s;
		e = f.e;

		mid_x = f.mid_x;
		mid_y = f.mid_y;
		mid_v = f.mid_v;

		bb_xs = f.bb_xs;
		bb_xe = f.bb_xe;

		bb_ys = f.bb_ys;
		bb_ye = f.bb_ye;

		es_x = f.es_x;
		es_y = f.es_y;

		// Area
		Area = f.Area;

		// Normal vector
		N_x = f.N_x;
		N_y = f.N_y;

		// Existed id
		existed_id = f.existed_id;
	}
};

class myRectangle
{
public:
	// Coordinates of starting point
	float x;
	float y;

	float w;  // Width
	float h;  // Height

	CUDA_CALLABLE_MEMBER myRectangle()
	{
		x = 0.0;
		y = 0.0;
		w = 1.0;
		h = 1.0;
	}

	CUDA_CALLABLE_MEMBER myRectangle(float x0, float y0, float w0, float h0)
	{
		x = x0;
		y = y0;
		w = w0;
		h = h0;
	}

	CUDA_CALLABLE_MEMBER ~myRectangle() {}

	CUDA_CALLABLE_MEMBER void operator=(const myRectangle& v)
	{
		x = v.x;
		y = v.y;
		w = v.w;
		h = v.h;
	}

	CUDA_CALLABLE_MEMBER bool contains(vertex p)  // Whether vertex p is inside the rectangle
	{
		return (p.x > x&& p.x <= x + w && p.y > y&& p.y <= y + h);
	}

	CUDA_CALLABLE_MEMBER bool intersects(myRectangle r)  // Whether two rectangles intersect
	{
		if (r.x > x + w || r.x + r.w < x || r.y > y + h || r.y + r.h < y)
		{
			return false;
		}
		return true;
	}
};

class QuadTreeNode  // One node corresponds to one face (capacity = 1)
{
public:
	myRectangle boundary;  // QuadTreeNode corresponds to a rectangular region
	face my_face;
	int face_num;  // face number: 0 or 1
	bool divided;  // Whether it has been divided into 4 sub-regions

	// My QuadTree follows "left-child right-sibling" manner
	int left_child;  // ID of left child
	// int right_sibling;  // Right sibling is always fetched by (ID+1)

	CUDA_CALLABLE_MEMBER QuadTreeNode(myRectangle boundary0)
	{
		boundary = boundary0;

		face_num = 0;
		divided = false;

		left_child = -1;
	}

	CUDA_CALLABLE_MEMBER ~QuadTreeNode() {}
};

class QuadTree
{
public:
	std::vector<QuadTreeNode> node_arr;  // QuadTree is saved using a node array. 
	int node_num;  // Number of nodes contained
	int face_num;  // Number of faces contained

	CUDA_CALLABLE_MEMBER QuadTree(myRectangle boundary0)
	{
		node_arr.clear();
		node_num = 1;
		face_num = 0;
		node_arr.push_back(QuadTreeNode(boundary0));  // The first node
	}

	CUDA_CALLABLE_MEMBER ~QuadTree() {}

	CUDA_CALLABLE_MEMBER int insert(face p)  // Insert a face into the QuadTree
	{
		// Optimized version
		if (!node_arr[0].boundary.contains(p.mid_v))  // p.m_v is not contained inside boundary
		{
			return -1;
		}

		int node_id = 0;
		while (true)  // Search recursively
		{
			QuadTreeNode tmp_node = node_arr[node_id];
			if (tmp_node.face_num < 1)  // Inserted successfully
			{
				node_arr[node_id].my_face = p;
				node_arr[node_id].face_num += 1;
				face_num += 1;
				return node_id;
			}
			else  // Move by changing node_id
			{
				float sub_x = tmp_node.boundary.x;
				float sub_y = tmp_node.boundary.y;
				float sub_w = tmp_node.boundary.w;
				float sub_h = tmp_node.boundary.h;

				// Check if divided
				if (!tmp_node.divided)  // If not, need to divide
				{
					myRectangle ne = myRectangle(sub_x + 0.5 * sub_w, sub_y + 0.5 * sub_h,
						0.5 * sub_w, 0.5 * sub_h);
					// myQuadTreeNode northeast = myQuadTreeNode(ne);

					myRectangle nw = myRectangle(sub_x, sub_y + 0.5 * sub_h, 0.5 * sub_w,
						0.5 * sub_h);
					// myQuadTreeNode northwest = myQuadTreeNode(nw);

					myRectangle se = myRectangle(sub_x + 0.5 * sub_w, sub_y, 0.5 * sub_w,
						0.5 * sub_h);
					// myQuadTreeNode southeast = myQuadTreeNode(se);

					myRectangle sw = myRectangle(sub_x, sub_y, 0.5 * sub_w, 0.5 * sub_h);
					// myQuadTreeNode southwest = myQuadTreeNode(sw);

					node_arr[node_id].left_child = node_num;

					node_arr.push_back(QuadTreeNode(ne));

					node_arr.push_back(QuadTreeNode(nw));

					node_arr.push_back(QuadTreeNode(se));

					node_arr.push_back(QuadTreeNode(sw));

					node_num += 4;

					node_arr[node_id].divided = true;
				}

				//  Divided! Go check the 4 children
				//  Decide which child to go
				node_id = node_arr[node_id].left_child;

				if (p.mid_y <= sub_y + 0.5 * sub_h)
					node_id += 2;
				if (p.mid_x <= sub_x + 0.5 * sub_w)
					node_id++;
			}
		}
		return -1;
	}

	CUDA_CALLABLE_MEMBER void operator=(const QuadTree& qt)
	{
		node_num = qt.node_num;
		face_num = qt.face_num;

		node_arr.clear();

		node_arr.assign(qt.node_arr.begin(), qt.node_arr.end());
	}
};

#endif