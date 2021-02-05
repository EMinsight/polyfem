#include "MshReader.hpp"

#include <polyfem/Logger.hpp>
#include <polyfem/StringUtils.hpp>

#include <MshIO/mshio.h>

#include <fstream>
#include <string>
#include <iostream>
#include <vector>

namespace polyfem
{
	bool MshReader::load(const std::string &path, Eigen::MatrixXd &vertices, Eigen::MatrixXi &cells, std::vector<std::vector<int>> &elements, std::vector<std::vector<double>> &weights)
	{
		mshio::MshSpec spec = mshio::load_msh(path);
		const auto &nodes = spec.nodes;
		const int n_vertices = nodes.num_nodes;
		const int dim = nodes.entity_blocks[0].entity_dim;

		vertices.resize(n_vertices, dim);
		int index = 0;
		for (const auto &n : nodes.entity_blocks)
		{
			for (int i = 0; i < n.num_nodes_in_block * n.entity_dim; i += n.entity_dim)
			{
				if (dim == 2)
					vertices.row(index) << n.data[i], n.data[i + 1];
				if (dim == 3)
					vertices.row(index) << n.data[i], n.data[i + 1], n.data[i + 2];

				++index;
			}
		}

		const auto &els = spec.elements;

		int cells_cols = -1;
		int num_els = 0;
		for (const auto &e : els.entity_blocks)
		{
			const int type = e.element_type;

			if (type == 2 || type == 9 || type == 21 || type == 23 || type == 25) //tri
			{
				assert(cells_cols == -1);
				cells_cols = 3;
				num_els = e.num_elements_in_block;
			}
			else if (type == 3 || type == 10) //quad
			{
				assert(cells_cols == -1);
				cells_cols = 4;
				num_els = e.num_elements_in_block;
			}
			else if (type == 4 || type == 11 || type == 29 || type == 30 || type == 31) //tet
			{
				assert(cells_cols == -1);
				cells_cols = 4;
				num_els = e.num_elements_in_block;
			}
			else if (type == 5 || type == 12) //hex
			{
				assert(cells_cols == -1);
				cells_cols = 8;
				num_els = e.num_elements_in_block;
			}
		}
		assert(cells_cols > 0);

		cells.resize(num_els, cells_cols);
		elements.resize(num_els);
		weights.resize(num_els);
		for (const auto &e : els.entity_blocks)
		{
			const int type = e.element_type;
			if (type == 2 || type == 9 || type == 21 || type == 23 || type == 25 ||
				type == 3 || type == 10 ||
				type == 4 || type == 11 || type == 29 || type == 30 || type == 31 ||
				type == 5 || type == 12)
			{
				const size_t n_nodes = mshio::nodes_per_element(type);
				for (int i = 0; i < e.data.size(); i += (n_nodes + 1))
				{
					const int cell_index = e.data[i] - 1;
					index = 0;
					for (int j = i + 1; j <= i + cells_cols; ++j)
						cells(cell_index, index++) = e.data[j] - 1;

					for (int j = i + 1; j < i + 1 + n_nodes; ++j)
					{
						elements[cell_index].push_back(e.data[j] - 1);
					}
				}
			}
		}

		// std::ifstream infile(path.c_str());

		// std::string line;

		// int phase = -1;
		// int line_number = -1;
		// bool size_read = false;

		// int n_triangles = 0;
		// int n_tets = 0;

		// std::vector<std::vector<double>> all_elements;

		// while (std::getline(infile, line))
		// {
		// 	line = StringUtils::trim(line);
		// 	++line_number;

		// 	if (line.empty())
		// 		continue;

		// 	if (line[0] == '$')
		// 	{
		// 		if (line.substr(1, 3) == "End")
		// 			phase = -1;
		// 		else
		// 		{
		// 			const auto header = line.substr(1);

		// 			if (header.find("MeshFormat") == 0)
		// 				phase = 0;
		// 			else if (header.find("Nodes") == 0)
		// 				phase = 1;
		// 			else if (header.find("Elements") == 0)
		// 				phase = 2;
		// 			else
		// 			{
		// 				logger().debug("{}: [Warning] ignoring {}", line_number, header);
		// 				phase = -1;
		// 			}
		// 		}

		// 		size_read = false;

		// 		continue;
		// 	}

		// 	if (phase == -1)
		// 		continue;

		// 	std::istringstream iss(line);
		// 	//header
		// 	if (phase == 0)
		// 	{
		// 		double version_number;
		// 		int file_type;
		// 		int data_size;

		// 		iss >> version_number >> file_type >> data_size;

		// 		assert(version_number == 2.2);
		// 		assert(file_type == 0);
		// 		assert(data_size == 8);
		// 	}
		// 	//coordiantes
		// 	else if (phase == 1)
		// 	{
		// 		if (!size_read)
		// 		{
		// 			int n_vertices;
		// 			iss >> n_vertices;
		// 			vertices.resize(n_vertices, 3);
		// 			size_read = true;
		// 		}
		// 		else
		// 		{
		// 			int node_number;
		// 			double x_coord, y_coord, z_coord;

		// 			iss >> node_number >> x_coord >> y_coord >> z_coord;
		// 			//node_numbers starts with 1
		// 			vertices.row(node_number - 1) << x_coord, y_coord, z_coord;
		// 		}
		// 	}
		// 	//elements
		// 	else if (phase == 2)
		// 	{
		// 		if (!size_read)
		// 		{
		// 			int number_of_elements;
		// 			iss >> number_of_elements;
		// 			all_elements.resize(number_of_elements);
		// 			size_read = true;
		// 		}
		// 		else
		// 		{
		// 			int elm_number, elm_type, number_of_tags;

		// 			iss >> elm_number >> elm_type >> number_of_tags;

		// 			//9-node third order incomplete triangle
		// 			assert(elm_type != 20);

		// 			//12-node fourth order incomplete triangle
		// 			assert(elm_type != 22);

		// 			//15-node fifth order incomplete triangle
		// 			assert(elm_type != 24);

		// 			//21-node fifth order complete triangle
		// 			assert(elm_type != 25);

		// 			//56-node fifth order tetrahedron
		// 			assert(elm_type != 31);

		// 			//60 is the new rational element
		// 			if (elm_type == 2 || elm_type == 9 || elm_type == 21 || elm_type == 23 || elm_type == 60)
		// 				++n_triangles;
		// 			else if (elm_type == 4 || elm_type == 11 || elm_type == 29 || elm_type == 30)
		// 				++n_tets;

		// 			//skipping tags
		// 			for (int i = 0; i < number_of_tags; ++i)
		// 			{
		// 				int tmp;
		// 				iss >> tmp;
		// 			}

		// 			auto &node_list = all_elements[elm_number - 1];
		// 			node_list.push_back(elm_type);

		// 			while (iss.good())
		// 			{
		// 				double tmp;
		// 				iss >> tmp;
		// 				node_list.push_back(tmp);
		// 			}
		// 		}
		// 	}
		// 	else
		// 	{
		// 		assert(false);
		// 	}
		// }

		// int index = 0;
		// if (n_tets == 0)
		// {
		// 	elements.resize(n_triangles);
		// 	weights.resize(n_triangles);
		// 	cells.resize(n_triangles, 3);

		// 	for (const auto &els : all_elements)
		// 	{
		// 		const int elm_type = els[0];
		// 		if (elm_type != 2 && elm_type != 9 && elm_type != 21 && elm_type != 23 && elm_type != 60)
		// 			continue;

		// 		auto &el = elements[index];
		// 		auto &wh = weights[index];
		// 		for (size_t i = 1; i < (elm_type == 60 ? 7 : els.size()); ++i)
		// 			el.push_back(int(els[i]) - 1);
		// 		if (elm_type == 60)
		// 		{
		// 			for (size_t i = 7; i < els.size(); ++i)
		// 				wh.push_back(els[i]);

		// 			assert(wh.size() == el.size());
		// 			assert(wh.size() == 6);
		// 		}

		// 		cells.row(index) << el[0], el[1], el[2];

		// 		++index;
		// 	}
		// }
		// else
		// {
		// 	elements.resize(n_tets);
		// 	weights.resize(n_tets);
		// 	cells.resize(n_tets, 4);

		// 	for (const auto &els : all_elements)
		// 	{
		// 		const int elm_type = els[0];
		// 		if (elm_type != 4 && elm_type != 11 && elm_type != 29 && elm_type != 30)
		// 			continue;

		// 		auto &el = elements[index];
		// 		auto &wh = weights[index];
		// 		for (size_t i = 1; i < els.size(); ++i)
		// 			el.push_back(els[i] - 1);

		// 		cells.row(index) << el[0], el[1], el[2], el[3];
		// 		++index;
		// 	}
		// }

		return true;
	}
} // namespace polyfem