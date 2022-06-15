#include <polyfem/State.hpp>

#include <polyfem/utils/FEBioReader.hpp>

#include <polyfem/mesh/GeometryReader.hpp>
#include <polyfem/mesh/mesh2D/CMesh2D.hpp>
#include <polyfem/mesh/mesh2D/NCMesh2D.hpp>
#include <polyfem/mesh/mesh3D/Mesh3D.hpp>

#include <polyfem/utils/Selection.hpp>

#include <polyfem/utils/JSONUtils.hpp>

#include <igl/Timer.h>
namespace polyfem
{
	using namespace basis;
	using namespace mesh;
	using namespace utils;

	void State::reset_mesh()
	{
		bases.clear();
		pressure_bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();
		parent_elements.clear();
		obstacle.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;
	}

	void State::load_mesh(GEO::Mesh &meshin, const std::function<int(const RowVectorNd &)> &boundary_marker, bool non_conforming, bool skip_boundary_sideset)
	{
		reset_mesh();

		igl::Timer timer;
		timer.start();
		logger().info("Loading mesh...");
		mesh = Mesh::create(meshin, non_conforming);
		if (!mesh)
		{
			logger().error("Unable to load the mesh");
			return;
		}

		// TODO: renable this
		// if (args["normalize_mesh"])
		// 	mesh->normalize();
		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{} {}], max [{} {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{} {} {}], max [{} {} {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		assembler.set_size(mesh->dimension());

		// TODO: renable this
		// int n_refs = args["n_refs"];
		// if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		// {
		// 	if (args["force_no_ref_for_harmonic"])
		// 		logger().warn("Using harmonic bases without refinement");
		// 	else
		// 		n_refs = 1;
		// }
		// if (n_refs > 0)
		// 	mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		// TODO: renable this
		// if (!skip_boundary_sideset)
		// 	mesh->compute_boundary_ids(boundary_marker);
		// BoxSetter::set_sidesets(args, *mesh);
		set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
			assembler.init_multimaterial(mesh->is_volume(), Es, nus);
			density.init_multimaterial(rhos);
		});

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		timer.start();
		logger().info("Loading obstacles...");
		mesh::read_obstacle_geometry(
			args["geometry"], args["boundary_conditions"]["obstacle_displacements"],
			args["root_path"], mesh->dimension(), obstacle);
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		ref_element_sampler.init(mesh->is_volume(), mesh->n_elements(), args["output"]["paraview"]["vismesh_rel_area"]);
	}

	void State::load_mesh(bool non_conforming,
						  const std::vector<std::string> &names,
						  const std::vector<Eigen::MatrixXi> &cells,
						  const std::vector<Eigen::MatrixXd> &vertices)
	{
		assert(names.size() == cells.size());
		assert(vertices.size() == cells.size());

		reset_mesh();

		igl::Timer timer;
		timer.start();

		logger().info("Loading mesh ...");
		if (mesh == nullptr)
		{
			assert(is_param_valid(args, "geometry"));
			mesh::read_fem_geometry(
				args["geometry"], args["root_path"], mesh,
				names, vertices, cells, non_conforming);
		}

		if (mesh == nullptr)
		{
			logger().error("unable to load the mesh!");
			return;
		}

		// if(!flipped_elements.empty())
		// {
		// 	mesh->compute_elements_tag();
		// 	for(auto el_id : flipped_elements)
		// 		mesh->set_tag(el_id, ElementType::InteriorPolytope);
		// }

		RowVectorNd min, max;
		mesh->bounding_box(min, max);

		if (min.size() == 2)
			logger().info("mesh bb min [{}, {}], max [{}, {}]", min(0), min(1), max(0), max(1));
		else
			logger().info("mesh bb min [{}, {}, {}], max [{}, {}, {}]", min(0), min(1), min(2), max(0), max(1), max(2));

		assembler.set_size(mesh->dimension());

		// if (n_refs <= 0 && args["poly_bases"] == "MFSHarmonic" && mesh->has_poly())
		// {
		// 	if (args["force_no_ref_for_harmonic"])
		// 		logger().warn("Using harmonic bases without refinement");
		// 	else
		// 		n_refs = 1;
		// }

		// if (n_refs > 0)
		// 	mesh->refine(n_refs, args["refinenemt_location"], parent_elements);

		// mesh->set_tag(1712, ElementType::InteriorPolytope);

		// double boundary_id_threshold = args["boundary_id_threshold"];
		// if (boundary_id_threshold <= 0)
		// 	boundary_id_threshold = mesh->is_volume() ? 1e-2 : 1e-7;

		// if (!mesh->has_boundary_ids())
		// {
		// 	const std::string bc_tag_path = resolve_input_path(args["bc_tag"]);
		// 	if (bc_tag_path.empty())
		// 		mesh->compute_boundary_ids(boundary_id_threshold);
		// 	else
		// 		mesh->load_boundary_ids(bc_tag_path);
		// }
		// BoxSetter::set_sidesets(args, *mesh);
		set_multimaterial([&](const Eigen::MatrixXd &Es, const Eigen::MatrixXd &nus, const Eigen::MatrixXd &rhos) {
			assembler.init_multimaterial(mesh->is_volume(), Es, nus);
			density.init_multimaterial(rhos);
		});

		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());

		ref_element_sampler.init(mesh->is_volume(), mesh->n_elements(), args["output"]["paraview"]["vismesh_rel_area"]);

		timer.start();
		logger().info("Loading obstacles...");
		mesh::read_obstacle_geometry(
			args["geometry"], args["boundary_conditions"]["obstacle_displacements"],
			args["root_path"], mesh->dimension(), obstacle, names, vertices, cells);
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());
	}

	void State::load_febio(const std::string &path, const json &args_in)
	{
		FEBioReader::load(path, args_in, *this);

		igl::Timer timer;
		timer.start();
		logger().info("Loading obstacles...");
		mesh::read_obstacle_geometry(
			args["geometry"], args["boundary_conditions"]["obstacle_displacements"],
			args["root_path"], mesh->dimension(), obstacle);
		timer.stop();
		logger().info(" took {}s", timer.getElapsedTime());
	}

	void State::compute_mesh_stats()
	{
		if (!mesh)
		{
			logger().error("Load the mesh first!");
			return;
		}

		bases.clear();
		pressure_bases.clear();
		geom_bases.clear();
		boundary_nodes.clear();
		local_boundary.clear();
		local_neumann_boundary.clear();
		polys.clear();
		poly_edge_to_data.clear();

		stiffness.resize(0, 0);
		rhs.resize(0, 0);
		sol.resize(0, 0);
		pressure.resize(0, 0);

		n_bases = 0;
		n_pressure_bases = 0;

		simplex_count = 0;
		regular_count = 0;
		regular_boundary_count = 0;
		simple_singular_count = 0;
		multi_singular_count = 0;
		boundary_count = 0;
		non_regular_boundary_count = 0;
		non_regular_count = 0;
		undefined_count = 0;
		multi_singular_boundary_count = 0;

		const auto &els_tag = mesh->elements_tag();

		mesh->prepare_mesh();

		for (size_t i = 0; i < els_tag.size(); ++i)
		{
			const ElementType type = els_tag[i];

			switch (type)
			{
			case ElementType::Simplex:
				simplex_count++;
				break;
			case ElementType::RegularInteriorCube:
				regular_count++;
				break;
			case ElementType::RegularBoundaryCube:
				regular_boundary_count++;
				break;
			case ElementType::SimpleSingularInteriorCube:
				simple_singular_count++;
				break;
			case ElementType::MultiSingularInteriorCube:
				multi_singular_count++;
				break;
			case ElementType::SimpleSingularBoundaryCube:
				boundary_count++;
				break;
			case ElementType::InterfaceCube:
			case ElementType::MultiSingularBoundaryCube:
				multi_singular_boundary_count++;
				break;
			case ElementType::BoundaryPolytope:
				non_regular_boundary_count++;
				break;
			case ElementType::InteriorPolytope:
				non_regular_count++;
				break;
			case ElementType::Undefined:
				undefined_count++;
				break;
			}
		}

		logger().info("simplex_count: \t{}", simplex_count);
		logger().info("regular_count: \t{}", regular_count);
		logger().info("regular_boundary_count: \t{}", regular_boundary_count);
		logger().info("simple_singular_count: \t{}", simple_singular_count);
		logger().info("multi_singular_count: \t{}", multi_singular_count);
		logger().info("boundary_count: \t{}", boundary_count);
		logger().info("multi_singular_boundary_count: \t{}", multi_singular_boundary_count);
		logger().info("non_regular_count: \t{}", non_regular_count);
		logger().info("non_regular_boundary_count: \t{}", non_regular_boundary_count);
		logger().info("undefined_count: \t{}", undefined_count);
		logger().info("total count:\t {}", mesh->n_elements());
	}

} // namespace polyfem
