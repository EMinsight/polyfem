#pragma once

#include "Parameter.hpp"
#include <igl/slim.h>

#include <ipc/collisions/collision_constraint.hpp>
#include <ipc/broad_phase/broad_phase.hpp>
#include "constraints/ShapeConstraints.hpp"

namespace polyfem
{
	class ShapeParameter : public Parameter
	{
	public:
		ShapeParameter(std::vector<std::shared_ptr<State>> &states_ptr, const json &args);

		void update() override
		{
		}

		Eigen::VectorXd initial_guess() const override
		{
			Eigen::VectorXd reduced;
			shape_constraints_->full_to_reduced(V_rest, reduced);
			return reduced;
		}

		Eigen::MatrixXd map(const Eigen::VectorXd &x) const override;
		Eigen::VectorXd map_grad(const Eigen::VectorXd &x, const Eigen::VectorXd &full_grad) const override;

		void smoothing(const Eigen::VectorXd &x, Eigen::VectorXd &new_x) override;
		bool is_step_valid(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		bool is_intersection_free(const Eigen::VectorXd &x) override;
		bool is_step_collision_free(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		double max_step_size(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;

		void line_search_begin(const Eigen::VectorXd &x0, const Eigen::VectorXd &x1) override;
		void line_search_end() override;
		void post_step(const int iter_num, const Eigen::VectorXd &x0) override;

		bool pre_solve(const Eigen::VectorXd &newX) override;
		void post_solve(const Eigen::VectorXd &newX) override;

		bool remesh(Eigen::VectorXd &x) override;
		void build_active_nodes();
		void build_tied_nodes(); // not applied to shape constraints

		std::map<int, std::vector<int>> optimization_boundary_to_node;

		const json &get_shape_params() const { return shape_params; }

	private:
		int iter = 0;
		int dim;

		Eigen::MatrixXd V_rest;
		Eigen::MatrixXi elements;

		std::vector<bool> active_nodes_mask;
		std::vector<bool> free_dimension; // not applied to shape constraints

		std::vector<bool> tied_nodes_mask; // not applied to shape constraints
		std::vector<std::array<int, 2>> tied_nodes; // not applied to shape constraints

		bool mesh_flipped = false;

		json shape_params, slim_params;

		// below only used for problems with contact

		bool has_collision;

		double _dhat;
		double _prev_distance;

		ipc::BroadPhaseMethod _broad_phase_method;
		double _ccd_tolerance;
		int _ccd_max_iterations;

		ipc::Constraints _constraint_set;
		ipc::CollisionMesh collision_mesh;
		ipc::Candidates _candidates;
		bool _use_cached_candidates = false;

		std::unique_ptr<ShapeConstraints> shape_constraints_;

		void update_constraint_set(const Eigen::MatrixXd &displaced_surface);
	};
} // namespace polyfem