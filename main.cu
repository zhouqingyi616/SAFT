#include <iostream>
#include <vector>
#include <algorithm>

#include <random>

#include <thrust/tuple.h>

#include <chrono>
#include <cassert>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"

#include "util_func.h"
#include "util_func_CUDA.h"
#include "shape_def.h"
#include "util_class.h"

#define micro_time_const 1e6  // 1 second = 1e6 microseconds

using namespace std::chrono;  // In order to get timing results

typedef std::chrono::high_resolution_clock Clock;

// Number of threads (denoted as "P" in the paper). 
// P=2048 works well for our Geforce RTX 2060. 
const int ELEM_NUM_PER_ITER = 2048;

// Number of threads per block
const int THREADS_PER_BLOCK_CONST = 128;  // 128 by default

// Number of threads used when calculating intersection. 
// See section 2.3, "Complexity of Calculate intersection" part for more details. 
const int MAX_THREAD_NUM = ELEM_NUM_PER_ITER * 8;  // Change it based on different GPU

// gpuAssert: gives information when CUDA is not working correctly. 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s, %s %s %d\n", cudaGetErrorName(code),
			cudaGetErrorString(code), file, line);
		if (abort) printf("CUDA not working properly\n");
	}
}

// Important functions used in main() function

// gpu_grow(): this is the "find new elements" step in paper. 
// Input: front_tree, id_arr
//   QuadTree (array of QuadTreeNode), as well as an array containing IDs to grow from.
// Output: s1_result, s2_result
//   save all the generated new faces
// See section 2.2.2 for more details. 
__global__ void gpu_grow(QuadTreeNode* front_tree, int* id_arr, int len_grow_id,
	face* s1_result, face* s2_result, int thread_num, int L_mid)
{
	// Calculate global thread ID
	int id_x = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (id_x < thread_num && id_x < len_grow_id)  // One thread handles one face
	{
		face add_face_arr[ADD_FACE_MAX_NUM];
		int add_id_arr[ADD_FACE_MAX_NUM];
		vertex front_i_arr[MID_NUM];
		vertex front_c_arr[ADD_C_MAX_NUM];

		int elem_id = id_x;

		// Initialize: area (length) of new face is -1.0
		s1_result[elem_id].Area = -1;
		s2_result[elem_id].Area = -1;

		int node_id = id_arr[elem_id];  // Fetch the corresponding QuadTreeNode
		QuadTreeNode tmp_node = front_tree[node_id];
		if (!(tmp_node.face_num == 0))  // This node contains a face
		{
			face tmp_grow_face = tmp_node.my_face;

			float tmp_elem_size = cuda_elem_size(tmp_grow_face.mid_x, tmp_grow_face.mid_y);

			float new_triangle_height = cuda_triangle_height(tmp_grow_face.Area, tmp_elem_size);

			// The ideal position, denoted as "c_i" in the paper. 
			// See section 2.2.2 for more details. 
			vertex tmp_ideal_center = vertex((tmp_grow_face.mid_x + new_triangle_height *
				tmp_grow_face.N_x), (tmp_grow_face.mid_y + new_triangle_height *
					tmp_grow_face.N_y));

			// Need to collect all faces which fall into a circular region. 
			// Calculate the radius of this circle
			float get_face_radius = 1.5 * (tmp_grow_face.Area + (tmp_elem_size - tmp_grow_face.Area)
				* int(tmp_elem_size > tmp_grow_face.Area));

			// Collect all the faces, save them in add_face_arr
			int add_face_num = cuda_get_faces_in_radius(elem_id, tmp_ideal_center,
				get_face_radius,
				front_tree,
				add_face_arr,
				add_id_arr);

			// Find all vertex candidates (from existing vertices) inside a circle
			// Save these candidates in front_c_arr
			int add_c_num = cuda_get_potent_c(elem_id,
				add_face_arr,
				add_face_num,
				tmp_grow_face,
				tmp_ideal_center,
				1.8 * new_triangle_height,
				front_c_arr);

			// Find the best candidate "optim_c" from front_c_arr. 
			auto optim_c_result = cuda_find_good_c(elem_id,
				front_c_arr,
				add_c_num,
				tmp_grow_face,
				tmp_elem_size,
				add_face_arr,
				add_face_num);

			vertex optim_c = thrust::get<0>(optim_c_result);
			float optim_c_quality = thrust::get<1>(optim_c_result);  // Calculated element quality
			bool c_exist = thrust::get<2>(optim_c_result);

			// Find all vertex candidates on perpendicular bisector
			// Save these candidates in front_i_arr
			cuda_get_potent_i(elem_id,
				L_mid,
				tmp_grow_face,
				tmp_ideal_center,
				front_i_arr);

			// Find the best candidate "optim_i" from front_i_arr.
			auto optim_i_result = cuda_find_good_i(elem_id,
				L_mid,
				front_i_arr,
				tmp_grow_face,
				tmp_elem_size,
				add_face_arr,
				add_face_num);

			vertex optim_i = thrust::get<0>(optim_i_result);
			float optim_i_quality = thrust::get<1>(optim_i_result);
			bool i_exist = thrust::get<2>(optim_i_result);

			// Choose the best candidate "new_vertex" between optim_c and optim_i
			auto new_vertex_result = cuda_choose_c_i(c_exist,
				i_exist,
				optim_c_quality,
				optim_i_quality,
				optim_c,
				optim_i);

			vertex new_vertex = thrust::get<0>(new_vertex_result);
			bool find_new_vertex = thrust::get<1>(new_vertex_result);
			bool is_existing_vertex = thrust::get<2>(new_vertex_result);

			// Record whether the new vertex coincides with some existing vertex
			new_vertex.if_mid = true;
			if (find_new_vertex && is_existing_vertex)
			{
				new_vertex.if_mid = false;
			}
			// Two new faces
			face s1 = face(tmp_grow_face.s, new_vertex);
			face s2 = face(new_vertex, tmp_grow_face.e);

			if (find_new_vertex)
			{
				// Normal vector of new faces need to point "outwards". 
				// Thus, invert normal vector direction of new faces if necessary
				float s1_change_dir = 1.0 - 2 * int(cuda_same_dir(s1, tmp_grow_face.e));
				s1.N_x *= s1_change_dir;
				s1.N_y *= s1_change_dir;

				float s2_change_dir = 1.0 - 2 * int(cuda_same_dir(s2, tmp_grow_face.s));
				s2.N_x *= s2_change_dir;
				s2.N_y *= s2_change_dir;

				if (is_existing_vertex)  // The candidate is some existing vertex
				{
					// Check whether new faces coincide with some existing face
					// Check s1 first
					auto find_s1_result = cuda_if_existing_face(elem_id,
						s1, add_face_arr, add_id_arr, add_face_num);

					bool s1_in = thrust::get<0>(find_s1_result);
					int s1_id = thrust::get<1>(find_s1_result);

					if (s1_in)
						s1.existed_id = s1_id;  // Record the corresponding ID

					// Then check s2
					auto find_s2_result = cuda_if_existing_face(elem_id,
						s2, add_face_arr, add_id_arr, add_face_num);

					bool s2_in = thrust::get<0>(find_s2_result);
					int s2_id = thrust::get<1>(find_s2_result);

					if (s2_in)
						s2.existed_id = s2_id;  // Record the corresponding ID
				}
				// Fill the results into array: s1_result & s2_result. 
				s1_result[elem_id] = s1;
				s2_result[elem_id] = s2;
			}
			else  // No legal candidate can be found: keep area (length) as -1.0
			{
				s1.Area = -1;
				s2.Area = -1;
				s1_result[elem_id] = s1;
				s2_result[elem_id] = s2;
			}
		}
	}
	return;
}

// gpu_intersect_fast(): this is the "calculate intersection" step in paper. 
// Input: s1_result, s2_result
//   arrays that contain all new faces generated by N threads.
// Output: inter_result, keep_result
//   "inter_result" records whether i-th element intersects with j-th element;
//   "keep_result" records whether i-th element intersects with some e_j (j > i). 
//     This information is utilized in "choose legal elements" step to improve speed. 
// 
// See section 2.2.3 & 2.2.4 for more details. 
__global__ void gpu_intersect(face* s1_result, face* s2_result, int n, int N, int* keep_result,
	bool* inter_result, int P)
{
	// N is the actual number of elements,
	// while P is the number of threads used in this step. 
	int unit_per_thread = (N * N + P - 1) / P;  // How many entries should one thread calculate

	// Calculate global thread ID
	int id_thread = (blockIdx.x * blockDim.x) + threadIdx.x;

	// This thread works from "id_start"-th entry to "id_end"-th entry.
	int id_start = id_thread * unit_per_thread;
	int id_end = id_start + unit_per_thread;

	for (int p = id_start; p < id_end; p++)
	{
		// Convert to id_x, id_y
		int id_x = p / N;
		int id_y = p % N;

		// Assume id_y > id_x, since the results are symmetric for id_y < id_x. 
		if (id_x < n && id_y < n && id_y > id_x)
		{
			face id_x_s1 = s1_result[id_x];
			face id_y_s1 = s1_result[id_y];

			// If distant, then they do not intersect; continue. 
			if ((id_x_s1.mid_x - id_y_s1.mid_x) * (id_x_s1.mid_x - id_y_s1.mid_x) +
				(id_x_s1.mid_y - id_y_s1.mid_y) * (id_x_s1.mid_y - id_y_s1.mid_y) > inter_max_dist_square)
				continue;

			face id_x_s2 = s2_result[id_x];
			face id_y_s2 = s2_result[id_y];

			// Distance threshold between two bounding boxes
			float bbox_diff = 0.5 * default_elem_size;

			// The bounding box of element id_x
			float id_x_xs = min(id_x_s1.bb_xs, id_x_s2.bb_xs);
			float id_x_xe = max(id_x_s1.bb_xe, id_x_s2.bb_xe);

			float id_x_ys = min(id_x_s1.bb_ys, id_x_s2.bb_ys);
			float id_x_ye = max(id_x_s1.bb_ye, id_x_s2.bb_ye);

			// The bounding box of element id_y
			float id_y_xs = min(id_y_s1.bb_xs, id_y_s2.bb_xs);
			float id_y_xe = max(id_y_s1.bb_xe, id_y_s2.bb_xe);

			float id_y_ys = min(id_y_s1.bb_ys, id_y_s2.bb_ys);
			float id_y_ye = max(id_y_s1.bb_ye, id_y_s2.bb_ye);

			// Check whether intersection happens
			// Here we check x and y direction separately
			if (!(id_x_xe < id_y_xs - bbox_diff || id_y_xe < id_x_xs - bbox_diff
				|| id_x_ye < id_y_ys - bbox_diff || id_y_ye < id_x_ys - bbox_diff))
			{
				if (keep_result[id_y] == 0)  // We use an "if" to avoid atomic operation
					keep_result[id_y] = 1;  // Intersect with element who has smaller id

				inter_result[id_y * N + id_x] = true;
			}
		}
	}
	return;
}

// Print some important GPU information
// Note that currently SAFT algorithm only works on single GPU. 
// We haven't implemented a multiple GPU version. 
void print_GPU_info()
{
	// First print out device properties
	int count;  // Number of visible GPUs
	cudaGetDeviceCount(&count);

	cudaDeviceProp prop;
	for (int i = 0; i < count; i++)  // Iterate over all GPU devices
	{
		cudaGetDeviceProperties(&prop, i);  // Fetch the device properties
		printf(" --- General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("Kernel execution timeout : ");

		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf(" --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);

		printf(" --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);  // 30 or 108
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
	return;
}

void main_double_ellipses()
{
	long long tmp_seed = time(0);  // Use current time as seed. 
	// long long tmp_seed = 1616464854;  // You can also set seed as a fixed number. 
	srand(tmp_seed);  // Set the seed for random number generator. 

	// Parameters: generals
	int threads_per_block = THREADS_PER_BLOCK_CONST;  // Number of threads per block. 
	int elem_num_per_iter = ELEM_NUM_PER_ITER;  // Thread number, denoted as "P" in the paper.

	// Define the problem size
	int p_mult = 14;  // Denoted as "p_mult" in the paper
	float r_mult = pow(2, 0.5 * p_mult);

	// Reconstruct QuadTree every "tree_recon_iter" iterations. 
	int tree_recon_iter = 20;  // See section 2.2.5 for more details. 
	// Maximum node number contained in the QuadTree
	// Estimate maximum node number ~ N(r) + P * T
	int max_node_num = 2000 * r_mult + elem_num_per_iter * tree_recon_iter * 10;

	printf("Max node num: %d\n", max_node_num);

	// Timing results initialization
	double avg_grow_time = 0.0;
	double avg_move_time = 0.0;
	double avg_intersect_time = 0.0;
	double avg_choose_time = 0.0;
	double avg_update_time = 0.0;

	// Parameters: mesh generation
	int L_mid = MID_NUM;  // Number of candidates on bisector

	// Parameters: initial geometry

	// In paper we test with "ring" case: double ellipses
	// with a_in = b_in = r_in, a_out = b_out = r_out.
	float r_in = 0.25 * r_mult;
	float r_out = 0.5 * r_mult;

	float S_domain = 0.75 * Pi * r_out * r_out;  // Area of the whole 2D domain
	float S_elem = (default_elem_size * default_elem_size) * sqrtf(3.0) / 4.0;
	int approx_elem_num = int(S_domain / S_elem);  // Estimate the number of elements

	// Parameters: plot
	bool if_plot = false;  // Whether you want to visualize the generated mesh
	bool if_print = true;  // Whether you want information been printed
	int max_iter = int(3 * approx_elem_num / elem_num_per_iter);  // Maximum number of iterations
	int plot_interval = 1 + int(max_iter / 20);  // Plot figure every "plot_interval" iterations
	int start_plot_iter = -1;  // Start plot after "start_plot_iter" iterations

	std::vector<int> empty_int_vec;  // Empty int vector, for future use

	std::vector<face> init_face_list = double_ellipses(r_in, r_in, r_out, r_out);
	int init_face_num = init_face_list.size();
	printf("Initial face num: %d\n", init_face_num);

	printf("Initializing...\n");
	myRectangle boundary = myRectangle(-1.1 * r_out, -1.1 * r_out,
		2.2 * r_out, 2.2 * r_out);  // Initialize QuadTree root node with rectangular region
	QuadTree front_tree = QuadTree(boundary);
	// Insert all initial fronts into the Quadtree "front_tree". 
	for (int i = 0; i < init_face_num; i++)
	{
		front_tree.insert(init_face_list[i]);
	}

	// Record archived faces
	std::vector<face> arch_face_list;  // Necessary if you want to visualize archived faces!

	// Allocate memory to save generated new faces: s1_results & s2_results
	// Allocate on GPU
	face* gpu_s1_result;
	face* gpu_s2_result;
	gpuErrchk(cudaMalloc(&gpu_s1_result, elem_num_per_iter * sizeof(face)));
	gpuErrchk(cudaMalloc(&gpu_s2_result, elem_num_per_iter * sizeof(face)));

	// Allocate on CPU
	std::vector<face> s1_result(elem_num_per_iter);
	std::vector<face> s2_result(elem_num_per_iter);

	// Allocate memory to save intersection results: intersect_arr
	// Allocate on GPU
	bool* intersect_arr_gpu;
	gpuErrchk(cudaMalloc(&intersect_arr_gpu, elem_num_per_iter * elem_num_per_iter * sizeof(bool)));

	// Allocate on CPU
	bool* intersect_arr;
	intersect_arr = (bool*)malloc(elem_num_per_iter * elem_num_per_iter * sizeof(bool));

	// Allocate memory to save intersection results: keep_arr
	// Allocate on GPU
	int* keep_arr_gpu;
	gpuErrchk(cudaMalloc(&keep_arr_gpu, elem_num_per_iter * sizeof(int)));

	// Allocate on CPU
	int* keep_arr;
	keep_arr = (int*)malloc(elem_num_per_iter * sizeof(int));

	// Allocate memory to save QuadTree (an array of QuadTreeNodes) on GPU
	QuadTreeNode* front_tree_gpu;
	gpuErrchk(cudaMalloc(&front_tree_gpu, max_node_num * sizeof(QuadTreeNode)));

	std::vector<int> arch_id_arr;

	int grow_iter = 0;
	int total_elem_num = 0;  // The total number of elements generated. 
	int total_thread_num_sum = 0;  // Sum up thread numbers (prepare for calculating utility). 

	printf("Entering loop...\n");

	auto loop_start = Clock::now();
	double loop_time = 0.0;
	while (grow_iter < max_iter)
	{
		// Whether information need to be printed during this iteration?
		bool should_print = if_print && grow_iter >= start_plot_iter &&
			grow_iter % plot_interval == 0;
		grow_iter += 1;

		if (should_print)
		{
			printf("\ngrow iter: %d / %d\n", grow_iter, max_iter);
			loop_time = (duration_cast<milliseconds>(Clock::now() - loop_start)).count();
			printf("Elapsed time: %.2f\n", loop_time / 1000.0);
			printf("Number of tree nodes: %d\n", front_tree.node_num);
		}

		// No active face left: the whole domain has been covered by triangular elements! 
		// Mesh generation finished successfully. 
		if (front_tree.face_num == 0)
		{
			printf("\nFinished successfully!\n");
			break;
		}

		int node_num = front_tree.node_num;
		if (node_num > max_node_num)
			printf("Node num exceeds limit! %d\n", node_num);

		int face_num = front_tree.face_num;
		if (should_print)
			printf("Remain front num: %d\n", face_num);

		// Determine the number of threads 
		int thread_num = elem_num_per_iter;
		if (50 < face_num && face_num < elem_num_per_iter)
		{
			thread_num = int(face_num / 2);  // Use fewer threads if there're not enough faces. 
		}
		else if (face_num <= 50)
		{
			thread_num = 1;  // Use only one thread when almost done. 
		}
		total_thread_num_sum += thread_num;

		// Calculate the number of blocks required. 
		int blocks_per_grid = (thread_num + threads_per_block - 1) / threads_per_block;

		// Start random sampling: which active faces should we grow from?
		std::vector<int> grow_id_arr;
		int len_grow_id_arr = 0;

		if (thread_num <= node_num)
		{
			grow_id_arr = fast_sample(front_tree, thread_num);
		}
		else  // There are more threads than nodes
		{
			grow_id_arr = fast_sample(front_tree, node_num);
		}
		len_grow_id_arr = grow_id_arr.size();

		// Visualize generated mesh if "if_plot == true". 
		if (grow_iter % plot_interval == 0 && grow_iter > start_plot_iter&& if_plot)
		{
			visualize_front(front_tree, arch_face_list, grow_id_arr, 50.0,
				-r_out * 1.1, r_out * 1.1, -r_out * 1.1, r_out * 1.1);
		}

		// Move data to GPU
		auto move_start = Clock::now();

		gpuErrchk(cudaMemcpy(front_tree_gpu, front_tree.node_arr.data(),
			node_num * sizeof(QuadTreeNode), cudaMemcpyHostToDevice));  // Move QuadTree

		int* grow_id_arr_gpu;
		gpuErrchk(cudaMalloc(&grow_id_arr_gpu, len_grow_id_arr * sizeof(int)));
		gpuErrchk(cudaMemcpy(grow_id_arr_gpu, grow_id_arr.data(),
			len_grow_id_arr * sizeof(int), cudaMemcpyHostToDevice));  // Move grow_id_arr

		avg_move_time += (duration_cast<microseconds>(Clock::now() - move_start)).count();

		// Grow new elements
		auto grow_start = Clock::now();

		gpu_grow << <blocks_per_grid, threads_per_block >> > (front_tree_gpu,
			grow_id_arr_gpu, len_grow_id_arr, gpu_s1_result, gpu_s2_result,
			thread_num, L_mid);

		cudaDeviceSynchronize();

		avg_grow_time += (duration_cast<microseconds>(Clock::now() - grow_start)).count();

		// Copy back data, from GPU to CPU. 
		move_start = Clock::now();

		gpuErrchk(cudaMemcpy(s1_result.data(), gpu_s1_result, elem_num_per_iter * sizeof(face),
			cudaMemcpyDeviceToHost));  // Copy back new face s1
		gpuErrchk(cudaMemcpy(s2_result.data(), gpu_s2_result, elem_num_per_iter * sizeof(face),
			cudaMemcpyDeviceToHost));  // copy back new face s2

		avg_move_time += (duration_cast<microseconds>(Clock::now() - move_start)).count();

		// Calculate intersection between new elements
		auto intersect_start = Clock::now();

		std::fill(keep_arr, keep_arr + elem_num_per_iter, 0);  // Initialize with all zeros. 

		gpuErrchk(cudaMemset(intersect_arr_gpu, 0, elem_num_per_iter *
			elem_num_per_iter * sizeof(bool)));  // cudaMemset: initialize with all zeros. 
		gpuErrchk(cudaMemset(keep_arr_gpu, 0, elem_num_per_iter * sizeof(int)));

		int blocks_inter_per_grid = ceil(float(MAX_THREAD_NUM) / threads_per_block);
		gpu_intersect << <blocks_inter_per_grid, threads_per_block >> >
			(gpu_s1_result, gpu_s2_result, len_grow_id_arr, elem_num_per_iter, keep_arr_gpu,
				intersect_arr_gpu, blocks_inter_per_grid * threads_per_block);

		cudaDeviceSynchronize();

		avg_intersect_time += (duration_cast<microseconds>(Clock::now() - intersect_start)).count();
		
		// Copy back data, from GPU to CPU. 
		move_start = Clock::now();
		gpuErrchk(cudaMemcpy(keep_arr, keep_arr_gpu, elem_num_per_iter * sizeof(int),
			cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(intersect_arr, intersect_arr_gpu, elem_num_per_iter *
			elem_num_per_iter * sizeof(bool), cudaMemcpyDeviceToHost));

		avg_move_time += (duration_cast<microseconds>(Clock::now() - move_start)).count();
		
		// Choose legal elements
		auto choose_start = Clock::now();

		std::vector<int> choose_arr(len_grow_id_arr);  // Decide: choose some new elements
		int new_elem_num = 0;

		// Use the slow & robust method only when number of remaining faces becomes small!
		if (front_tree.face_num < elem_num_per_iter)
		{
			for (int i = 0; i < len_grow_id_arr; ++i)  // i-th new element
			{
				if (s1_result[i].Area > 0 && s2_result[i].Area > 0)
				{
					// Check intersection
					float choose_inter_dot = inner_product(begin(choose_arr),
						begin(choose_arr) + i, intersect_arr + i * elem_num_per_iter, 0.0);

					// There's no intersection between i-th element and all chosen elements!
					if (choose_inter_dot < eps)
					{
						choose_arr[i] = 1;  // Get a legal element. 
						new_elem_num += 1;
					}
				}
			}
		}
		else  // Use the fast method (utilize "keel_arr") for most of the time
		{
			for (int i = 0; i < len_grow_id_arr; ++i)  // i-th new element
			{
				if (s1_result[i].Area > 0 && s2_result[i].Area > 0 && keep_arr[i] == 0)
				{
					choose_arr[i] = 1;  // Get a legal element. 
					new_elem_num += 1;
				}
				else
				{
					choose_arr[i] = 0;  // Discard the illegal element. 
				}
			}
		}

		avg_choose_time += (duration_cast<microseconds>(Clock::now() - choose_start)).count();
		
		if (should_print)
			printf("Generate new elem num: %d\n", new_elem_num);
		total_elem_num += new_elem_num;

		// Update active front
		auto update_start = Clock::now();

		// Go over all the new elements. 
		// New vertices / faces are recorded. 
		// This is the most time-consuming part in "update". 
		arch_id_arr.clear();

		for (int i = 0; i < len_grow_id_arr; i++)
		{
			if (choose_arr[i] == 1)  // A legal new element
			{
				vertex tmp_vertex = s1_result[i].e;
				arch_id_arr.push_back(grow_id_arr[i]);  // Archive grow face

				// Update face info

				// Update s1 info first
				if (s1_result[i].existed_id >= 0)  // Already existed front
				{
					arch_id_arr.push_back(s1_result[i].existed_id);
				}
				else  // New generated face
				{
					s1_result[i].e = tmp_vertex;
					s1_result[i].existed_id = -1;
					front_tree.insert(s1_result[i]);
				}

				// Update s2 info next
				if (s2_result[i].existed_id >= 0)
				{
					arch_id_arr.push_back(s2_result[i].existed_id);
				}
				else  // New generated face
				{
					s2_result[i].s = tmp_vertex;
					s2_result[i].existed_id = -1;
					front_tree.insert(s2_result[i]);
				}
			}
		}

		int arch_num = arch_id_arr.size();
		int new_node_num = front_tree.node_num;

		std::vector<bool> arch_bool(new_node_num);  // Record which face should be archived. 

		for (int i = 0; i < arch_num; i++)
		{
			int pop_id = arch_id_arr[i];
			if (pop_id < new_node_num && pop_id >= 0)
				arch_bool[pop_id] = true;
		}

		// Reconstruct the entire QuadTree, if necessary. 
		if ((grow_iter + 1) % tree_recon_iter == 0)
		{
			QuadTree new_front_tree(boundary);  // Create a new tree

			for (int i = 0; i < new_node_num; i++)  // Go over the existing QuadTree
			{
				if (front_tree.node_arr[i].face_num == 1)
				{
					if (!arch_bool[i])  // This active face must be included in the new QuadTree
					{
						new_front_tree.insert(front_tree.node_arr[i].my_face);
					}
					else  // Update the list of archived faces
					{
						
						arch_face_list.push_back(front_tree.node_arr[i].my_face);
					}
				}
			}
			front_tree = new_front_tree;
		}
		else  // For most of the time, we don't need to reconstruct the tree
		{
			for (int i = 0; i < new_node_num; i++)
			{
				if (arch_bool[i] && front_tree.node_arr[i].face_num == 1)  // Discard
				{
					// Update the list of archived faces
					// You can comment this if you do not want to visualize archived faces. 
					arch_face_list.push_back(front_tree.node_arr[i].my_face);

					front_tree.node_arr[i].face_num = 0;  // Mark the node as inactive. 
					front_tree.face_num -= 1;
				}
			}
		}

		// Don't forget to free CUDA memories allocated in this iteration
		cudaFree(grow_id_arr_gpu);

		avg_update_time += (duration_cast<microseconds>(Clock::now() - update_start)).count();
	}

	std::cout << "\n" << "Rand seed: " << tmp_seed << "\n";
	std::cout << "Total: " << total_elem_num << "\n";
	std::cout << "Thread utility: " << int(100.0 * total_elem_num /
		float(total_thread_num_sum)) << "\%" << "\n";
	std::cout << "Remain front: " << front_tree.face_num << "\n \n";
	

	std::cout << "total grow (s):" << avg_grow_time / micro_time_const << "\n";
	std::cout << "total intersect (s):" << avg_intersect_time / micro_time_const << "\n";
	std::cout << "total choose (s):" << avg_choose_time / micro_time_const << "\n";
	std::cout << "total update (s):" << avg_update_time / micro_time_const << "\n";
	std::cout << "total move (s):" << avg_move_time / micro_time_const << "\n";

	float total_time = (avg_grow_time + avg_move_time + avg_intersect_time +
		avg_choose_time + avg_update_time) / micro_time_const;
	std::cout << "total time (s):" << total_time << "\n";
	std::cout << "\n";

	return;
}

int main()
{
	print_GPU_info();

	main_double_ellipses();

	return 0;
}
