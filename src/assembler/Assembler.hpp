#ifndef ASSEMBLER_HPP
#define ASSEMBLER_HPP

#include <polyfem/ElementAssemblyValues.hpp>

#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>

namespace polyfem
{
	class StiffnessRawData
	{
	private:
		std::vector<int> outer_index;
		std::vector<int> inner_index;
		// std::vector<std::map<int, int>> index_mapping;
		std::vector<std::vector<std::pair<int, int>>> index_mapping;
		int size;

	public:
		void init(const StiffnessMatrix &mat)
		{
			size = mat.rows();
			index_mapping.resize(mat.rows());

			outer_index.resize(mat.rows() + 1);
			inner_index.resize(mat.nonZeros());

			for (int i = 0; i < outer_index.size(); ++i)
				outer_index[i] = mat.outerIndexPtr()[i];

			for (int i = 0; i < inner_index.size(); ++i)
				inner_index[i] = mat.innerIndexPtr()[i];
		}

		inline const int nnz() const { return inner_index.size(); }

		void add_index(const int row, const int col, const int index)
		{
			// index_mapping[row][col] = index;
			index_mapping[row].emplace_back(col, index);
		}

		int operator()(const int row, const int col) const
		{
			// return index_mapping[row].at(col);
			const auto &tmp = index_mapping[row];
			for (const auto &p : tmp)
				if (p.first == col)
					return p.second;

			return -1;
		}

		template <typename DataVect>
		void build_matrix(const DataVect &values, StiffnessMatrix &stiffness) const
		{
			stiffness = Eigen::Map<const StiffnessMatrix>(size, size, values.size(), &outer_index[0], &inner_index[0], &values[0]);
		}
	};

	class IndexAssembler
	{
	public:
		void assemble(
			const bool is_tensor,
			const bool is_volume,
			const int n_basis,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			StiffnessRawData &index_mapping) const;
	};

	template <class LocalAssembler>
	class Assembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int n_basis,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			StiffnessMatrix &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};

	template <class LocalAssembler>
	class MixedAssembler
	{
	public:
		void assemble(
			const bool is_volume,
			const int n_psi_basis,
			const int n_phi_basis,
			const std::vector<ElementBases> &psi_bases,
			const std::vector<ElementBases> &phi_bases,
			const std::vector<ElementBases> &gbases,
			StiffnessMatrix &stiffness) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

	private:
		LocalAssembler local_assembler_;
	};

	template <class LocalAssembler>
	class NLAssembler
	{
	public:
		void assemble_grad(
			const bool is_volume,
			const int n_basis,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const Eigen::MatrixXd &displacement,
			Eigen::MatrixXd &rhs) const;

		void assemble_hessian(
			const bool is_volume,
			const int n_basis,
			const bool project_to_psd,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const Eigen::MatrixXd &displacement,
			const StiffnessRawData &index_mapping,
			StiffnessMatrix &grad) const;

		double assemble(
			const bool is_volume,
			const std::vector<ElementBases> &bases,
			const std::vector<ElementBases> &gbases,
			const Eigen::MatrixXd &displacement) const;

		inline LocalAssembler &local_assembler() { return local_assembler_; }
		inline const LocalAssembler &local_assembler() const { return local_assembler_; }

		void clear_cache() {}

	private:
		LocalAssembler local_assembler_;
	};
} // namespace polyfem

#endif //ASSEMBLER_HPP
