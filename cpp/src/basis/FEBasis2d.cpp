////////////////////////////////////////////////////////////////////////////////
#include "FEBasis2d.hpp"
#include "QuadQuadrature.hpp"
#include <igl/viewer/Viewer.h>
#include <cassert>
#include <array>
////////////////////////////////////////////////////////////////////////////////

/*

Axes:
 y
 |
 o──x

Boundaries:
X axis: left/right
Y axis: bottom/top

Corner nodes:
v3──────x─────v2
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x┄┄┄┄┄┄x┄┄┄┄┄┄x
 │      ┆      │
 │      ┆      │
 │      ┆      │
v0──────x─────v1

v0 = (0, 0)
v1 = (1, 0)
v2 = (1, 1)
v3 = (0, 1)

Edge nodes:
 x─────e2──────x
 │      ┆      │
 │      ┆      │
 │      ┆      │
e3┄┄┄┄┄┄x┄┄┄┄┄e1
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x─────e0──────x

e0  = (0.5,   0)
e1  = (  1, 0.5)
e2  = (0.5,   1)
e3  = (  0, 0.5)

*/

namespace {

template<typename T>
Eigen::MatrixXd alpha(int i, T &t) {
	switch (i) {
		case 0: return (1-t);
		case 1: return t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dalpha(int i, T &t) {
	switch (i) {
		case 0: return -1+0*t;
		case 1: return 1+0*t;;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd theta(int i, T &t) {
	switch (i) {
		case 0: return (1 - t) * (1 - 2 * t);
		case 1: return 4 * t * (1 - t);
		case 2: return t * (2 * t - 1);
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

template<typename T>
Eigen::MatrixXd dtheta(int i, T &t) {
	switch (i) {
		case 0: return -3+4*t;
		case 1: return 4-8*t;
		case 2: return -1+4*t;
		default: assert(false);
	}
	throw std::runtime_error("Invalid index");
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 2>, 8> linear_quad_local_node = {{
	{{0, 0}}, // v0  = (0, 0)
	{{1, 0}}, // v1  = (1, 0)
	{{1, 1}}, // v2  = (1, 1)
	{{0, 1}}, // v3  = (0, 1)
}};

void linear_quad_basis_value(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];
	val = alpha(idx[0], u).array() * alpha(idx[1], v).array();
}

void linear_quad_basis_grad(const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val) {
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = linear_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dalpha(idx[0], u).array() * alpha(idx[1], v).array();
	val.col(1) = alpha(idx[0], u).array() * dalpha(idx[1], v).array();
}

// -----------------------------------------------------------------------------

constexpr std::array<std::array<int, 2>, 9> quadr_quad_local_node = {{
	{{0, 0}}, // v0  = (  0,   0)
	{{2, 0}}, // v1  = (  1,   0)
	{{2, 2}}, // v2  = (  1,   1)
	{{0, 2}}, // v3  = (  0,   1)
	{{1, 0}}, // e0  = (0.5,   0)
	{{2, 1}}, // e1  = (  1, 0.5)
	{{1, 2}}, // e2  = (0.5,   1)
	{{0, 1}}, // e3  = (  0, 0.5)
	{{1, 1}}, // f0  = (0.5, 0.5)
}};

// -----------------------------------------------------------------------------

poly_fem::Navigation::Index find_edge(const poly_fem::Mesh2D &mesh, int f, int v1, int v2) {
	std::array<int, 2> v = {{v1, v2}};
	std::sort(v.begin(), v.end());
	auto idx = mesh.get_index_from_face(f, 0);
	for (int lv = 0; lv < mesh.n_element_vertices(idx.face); ++lv) {
		std::array<int, 2> u;
		u[0] = idx.vertex;
		u[1] = mesh.switch_vertex(idx).vertex;
		std::sort(u.begin(), u.end());
		if (u == v) {
			return idx;
		}
		idx = mesh.next_around_face(idx);
	}
	throw std::runtime_error("Edge not found");
}

// -----------------------------------------------------------------------------

constexpr std::array<int, 4> local_edge_to_interface_flag = {{
	poly_fem::InterfaceData::BOTTOM_FLAG,
	poly_fem::InterfaceData::RIGHT_FLAG,
	poly_fem::InterfaceData::TOP_FLAG,
	poly_fem::InterfaceData::LEFT_FLAG,
}};

// -----------------------------------------------------------------------------

std::array<int, 4> linear_quad_local_to_global(const poly_fem::Mesh2D &mesh, int f) {
	assert(mesh.n_element_vertices(f) == 4);

	// Vertex nodes
	std::array<int, 4> l2g;
	for (int lv = 0; lv < 4; ++lv) {
		l2g[lv] = mesh.vertex_global_index(f, lv);
	}

	return l2g;
}

// -----------------------------------------------------------------------------

std::array<int, 9> quadr_quad_local_to_global(const poly_fem::Mesh2D &mesh, int f) {
	assert(mesh.n_element_vertices(f) == 4);

	int e_offset = mesh.n_pts();
	int f_offset = e_offset + mesh.n_edges();

	// Vertex nodes
	Eigen::Matrix<int, 4, 1> v;
	for (int lv = 0; lv < 4; ++lv) {
		v[lv] = mesh.vertex_global_index(f, lv);
	}

	// Edge nodes
	Eigen::Matrix<int, 4, 1> e;
	Eigen::Matrix<int, 4, 2> ev;
	ev.row(0) << v[0], v[1];
	ev.row(1) << v[1], v[2];
	ev.row(2) << v[2], v[3];
	ev.row(3) << v[3], v[0];
	for (int le = 0; le < e.rows(); ++le) {
		e[le] = find_edge(mesh, f, ev(le, 0), ev(le, 1)).edge;
	}

	// Local to global mapping of node indices
	std::array<int, 9> l2g;

	// Assign global ids to local nodes
	{
		int i = 0;
		for (int lv = 0; lv < v.rows(); ++lv) {
			l2g[i++] = v[lv];
		}
		for (int le = 0; le < e.rows(); ++le) {
			l2g[i++] = e_offset + e[le];
		}
		l2g[i++] = f_offset + f;
	}

	return l2g;
}

// -----------------------------------------------------------------------------

///
/// @brief      Compute the list of global nodes for the mesh. If discr_order is
///             1 then this is the same as the vertices of the input mesh. If
///             discr_order is 2, then nodes are inserted in the middle of each
///             simplex (edge, facet, cell), and nodes per elements are numbered
///             accordingly.
///
/// @param[in]  mesh               The input mesh
/// @param[in]  discr_order        The discretization order
/// @param[out] nodes              The nodes positions
/// @param[out] boundary_nodes     List of boundary node indices
/// @param[out] element_nodes_id   List of node indices per element
/// @param[out] local_boundary     Which facet of the element are on the
///                                boundary
/// @param[out] poly_edge_to_data  Data for edges at interface with polygons
///
void compute_nodes(
	const poly_fem::Mesh2D &mesh,
	const int discr_order,
	std::vector<Eigen::RowVector2d> &nodes,
	std::vector<int> &boundary_nodes,
	std::vector<std::vector<int> > &element_nodes_id,
	std::vector<poly_fem::LocalBoundary> &local_boundary,
	std::map<int, poly_fem::InterfaceData> &poly_edge_to_data)
{
	const int n_nodes = mesh.n_pts() + (discr_order > 1 ? mesh.n_edges() + mesh.n_elements() : 0);
	const int e_offset = mesh.n_pts();
	const int f_offset = e_offset + mesh.n_edges();
	Eigen::MatrixXd all_nodes(n_nodes, 2);
	std::vector<bool> is_boundary(n_nodes, false);
	std::vector<int> remapped_node(n_nodes, -1);

	// Step 1: Compute all node positions + node boundary tag
	{
		for (int v = 0; v < mesh.n_pts(); ++v) {
			all_nodes.row(v) = mesh.point(v);
			is_boundary[v] = mesh.is_boundary_vertex(v);
		}
		if (discr_order > 1) {
			Eigen::MatrixXd bary;
			mesh.edge_barycenters(bary);
			for (int e = 0; e < mesh.n_edges(); ++e) {
				all_nodes.row(e_offset + e) = bary.row(e);
				is_boundary[e_offset + e] = mesh.is_boundary_edge(e);
			}
			mesh.face_barycenters(bary);
			for (int f = 0; f < mesh.n_elements(); ++f) {
				all_nodes.row(f_offset + f) = bary.row(f);
				is_boundary[f_offset + f] = false;
			}
		}
	}

	nodes.clear();
	local_boundary.clear();
	local_boundary.resize(mesh.n_elements());
	element_nodes_id.resize(mesh.n_elements());

	// Step 2: Keep only read real nodes + compute parametric boundary tag
	for (int f = 0; f < mesh.n_elements(); ++f) {
		if (mesh.n_element_vertices(f) != 4) { continue; } // Skip polygons

		// Create remapped node array for element
		auto remap_nodes = [&] (auto l2g) {
			std::vector<int> res;
			res.reserve(l2g.size());
			for (int id : l2g) {
				if (remapped_node[id] < 0) {
					remapped_node[id] = nodes.size();
					Eigen::RowVector2d pos = all_nodes.row(id);
					nodes.push_back(pos);
					if (is_boundary[id]) {
						boundary_nodes.push_back(remapped_node[id]);
					}
				}
				res.push_back(remapped_node[id]);
			}
			return res;
		};
		if (discr_order == 1) {
			element_nodes_id[f] = remap_nodes(linear_quad_local_to_global(mesh, f));
		} else {
			element_nodes_id[f] = remap_nodes(quadr_quad_local_to_global(mesh, f));
		}

		// List of edges around the quad
		std::array<int, 4> e;
		{
			auto l2g = quadr_quad_local_to_global(mesh, f);
			for (int le = 0; le < 4; ++le) {
				e[le] = l2g[4+le] - e_offset;
			}
		}

		// Set boundary edges
		if (mesh.is_boundary_edge(e[3])) {
			local_boundary[f].set_left_edge_id(e[3]);
			local_boundary[f].set_left_boundary();
		}
		if (mesh.is_boundary_edge(e[1])) {
			local_boundary[f].set_right_edge_id(e[1]);
			local_boundary[f].set_right_boundary();
		}
		if (mesh.is_boundary_edge(e[0])) {
			local_boundary[f].set_bottom_edge_id(e[0]);
			local_boundary[f].set_bottom_boundary();
		}
		if (mesh.is_boundary_edge(e[2])) {
			local_boundary[f].set_top_edge_id(e[2]);
			local_boundary[f].set_top_boundary();
		}
	}

	// Step 3: Iterate over edges of polygons and compute interface weights
	for (int f = 0; f < mesh.n_elements(); ++f) {
		if (mesh.n_element_vertices(f) == 4) { continue; } // Skip quads

		auto index = mesh.get_index_from_face(f, 0);
		for (int lv = 0; lv < mesh.n_element_vertices(f); ++lv) {
			auto index2 = mesh.switch_face(index);
			if (index2.face >= 0) {
				// Opposite face is a quad, we need to set interface data
				int f2 = index2.face;
				assert(mesh.n_element_vertices(f2) == 4);
				auto abc = poly_fem::FEBasis2d::quadr_quad_edge_local_nodes(mesh, index2);
				poly_fem::InterfaceData data;
				data.element_id = index2.face;
				data.flag = local_edge_to_interface_flag[abc[1] - 4];
				if (discr_order == 2) {
					for (auto local_node : abc) {
						data.node_id.push_back(element_nodes_id[f2][local_node]);
					}
					data.local_indices.assign(abc.begin(), abc.end());
				} else {
					assert(discr_order == 1);
					auto ab = poly_fem::FEBasis2d::linear_quad_edge_local_nodes(mesh, index2);
					for (auto local_node : ab) {
						data.node_id.push_back(element_nodes_id[f2][local_node]);
					}
					data.local_indices.assign(ab.begin(), ab.end());
				}
				data.vals.assign(data.local_indices.size(), 1);
				poly_edge_to_data[index2.edge] = data;
			}
			index = mesh.next_around_face(index);
		}
	}
}

/*
Axes:
 y
 |
 o──x

Boundaries:
X axis: left/right
Y axis: bottom/top

Edge nodes:
 x─────e2──────x
 │      ┆      │
 │      ┆      │
 │      ┆      │
e3┄┄┄┄┄┄x┄┄┄┄┄e1
 │      ┆      │
 │      ┆      │
 │      ┆      │
 x─────e0──────x
*/

// -----------------------------------------------------------------------------

template<class InputIterator, class T>
	int find_index(InputIterator first, InputIterator last, const T& val)
{
	return std::distance(first, std::find(first, last, val));
}

} // anonymous namespace

////////////////////////////////////////////////////////////////////////////////

std::array<int, 2> poly_fem::FEBasis2d::linear_quad_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.n_element_vertices(f) == 4);

	// Local to global mapping of node indices
	auto l2g = linear_quad_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 2> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

std::array<int, 3> poly_fem::FEBasis2d::quadr_quad_edge_local_nodes(
	const Mesh2D &mesh, Navigation::Index index)
{
	int f = index.face;
	assert(mesh.n_element_vertices(f) == 4);
	int e_offset = mesh.n_pts();

	// Local to global mapping of node indices
	auto l2g = quadr_quad_local_to_global(mesh, f);

	// Extract requested interface
	std::array<int, 3> result;
	result[0] = find_index(l2g.begin(), l2g.end(), index.vertex);
	result[1] = find_index(l2g.begin(), l2g.end(), e_offset + index.edge);
	result[2] = find_index(l2g.begin(), l2g.end(), mesh.switch_vertex(index).vertex);
	return result;
}

namespace {

	Eigen::RowVector2d quadr_quad_local_node_coordinates(int local_index) {
		auto p = quadr_quad_local_node[local_index];
		return Eigen::RowVector2d(p[0], p[1]) / 2.0;
	}

} // anonymous namespace

Eigen::MatrixXd poly_fem::FEBasis2d::quadr_quad_edge_local_nodes_coordinates(
	const Mesh2D &mesh, Navigation::Index index)
{
	auto idx = quadr_quad_edge_local_nodes(mesh, index);
	Eigen::MatrixXd res(idx.size(), 2);
	int cnt = 0;
	for (int i : idx) {
		res.row(cnt++) = quadr_quad_local_node_coordinates(i);
	}
	return res;
}

// -----------------------------------------------------------------------------

void poly_fem::FEBasis2d::quadr_quad_basis_value(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];
	val = theta(idx[0], u).array() * theta(idx[1], v).array();
}

void poly_fem::FEBasis2d::quadr_quad_basis_grad(
	const int local_index, const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
{
	auto u=uv.col(0).array();
	auto v=uv.col(1).array();

	std::array<int, 2> idx = quadr_quad_local_node[local_index];

	val.resize(uv.rows(), 2);
	val.col(0) = dtheta(idx[0], u).array() * theta(idx[1], v).array();
	val.col(1) = theta(idx[0], u).array() * dtheta(idx[1], v).array();
}

// -----------------------------------------------------------------------------

int poly_fem::FEBasis2d::build_bases(
	const Mesh2D &mesh,
	const int quadrature_order,
	const int discr_order,
	std::vector<ElementBases> &bases,
	std::vector<LocalBoundary> &local_boundary,
	std::vector<int> &boundary_nodes,
	std::map<int, InterfaceData> &poly_edge_to_data)
{
	assert(!mesh.is_volume());

	std::vector<Eigen::RowVector2d> nodes;
	std::vector<std::vector<int>> element_nodes_id;
	compute_nodes(mesh, discr_order, nodes, boundary_nodes, element_nodes_id, local_boundary, poly_edge_to_data);

	QuadQuadrature quad_quadrature;
	bases.resize(mesh.n_elements());
	for (int e = 0; e < mesh.n_elements(); ++e) {
		ElementBases &b = bases[e];
		const int n_el_vertices = mesh.n_element_vertices(e);
		const int n_el_bases = (int) element_nodes_id[e].size();
		b.bases.resize(n_el_bases);

		if (n_el_vertices == 4) {
			quad_quadrature.get_quadrature(quadrature_order, b.quadrature);
			b.bases.resize(n_el_bases);

			for (int j = 0; j < n_el_bases; ++j) {
				const int global_index = element_nodes_id[e][j];

				b.bases[j].init(global_index, j, nodes[global_index]);

				if (discr_order == 1) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_quad_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ linear_quad_basis_grad(j, uv, val); });
				} else if (discr_order == 2) {
					b.bases[j].set_basis([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_quad_basis_value(j, uv, val); });
					b.bases[j].set_grad([j](const Eigen::MatrixXd &uv, Eigen::MatrixXd &val)
						{ quadr_quad_basis_grad(j, uv, val); });
				} else {
					assert(false);
				}
			}
		} else {
			// Polygonal bases are built later on
			// assert(false);
		}
	}

	return (int) nodes.size();
}
