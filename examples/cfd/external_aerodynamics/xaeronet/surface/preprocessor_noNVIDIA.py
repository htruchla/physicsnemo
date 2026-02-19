# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code processes mesh data from .stl and .vtp files to create partitioned
graphs for large scale training. It first converts meshes to triangular format
and extracts surface triangles, vertices, and relevant attributes such as pressure
and shear stress. Using nearest neighbors, the code interpolates these attributes
for a sampled boundary of points, and constructs a graph based on these points, with
node features like coordinates, normals, pressure, and shear stress, as well as edge
features representing relative displacement. The graph is partitioned into subgraphs,
and the partitions are saved. The code supports parallel processing to handle multiple
samples simultaneously, improving efficiency. Additionally, it provides an option to
save the point cloud of each graph for visualization purposes.
"""

import os
import vtk
import pyvista as pv
import numpy as np
import torch
import hydra

import torch_geometric as pyg

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

import importlib.util as _ilu

def _import_read_vtp():
    """
    Import read_vtp directly from its source file to avoid the broken
    physicsnemo.datapipes.cae.__init__.py, which unconditionally imports
    MeshDatapipe and therefore requires NVIDIA DALI even when it's not needed.
    """
    import physicsnemo as _pm
    _readers_path = os.path.join(
        os.path.dirname(_pm.__file__), "datapipes", "cae", "readers.py"
    )
    _spec = _ilu.spec_from_file_location("_physicsnemo_cae_readers", _readers_path)
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.read_vtp

read_vtp = _import_read_vtp()
class Tessellation:
    """
    Drop-in replacement for physicsnemo.sym.geometry.tessellation.Tessellation.
    Uses pyvista to load an STL and sample boundary points with normals and areas,
    returning the same dict structure that the original class produces.
    """

    def __init__(self, mesh: pv.PolyData):
        # Ensure triangulated surface with cell normals and areas pre-computed
        self._mesh = mesh.triangulate().compute_normals(
            cell_normals=True, point_normals=False
        )
        # Per-triangle area (pyvista stores it after compute_cell_sizes)
        sized = self._mesh.compute_cell_sizes(area=True, length=False, volume=False)
        self._areas = sized.cell_data["Area"]          # shape (n_triangles,)
        self._normals = self._mesh.cell_data["Normals"] # shape (n_triangles, 3)
        self._total_area = self._areas.sum()

    @classmethod
    def from_stl(cls, stl_path: str, airtight: bool = False):
        mesh = pv.read(stl_path)
        return cls(mesh)

    def sample_boundary(self, num_points: int) -> dict:
        """
        Sample `num_points` points on the surface, weighted by triangle area.
        Returns a dict with keys x, y, z, normal_x, normal_y, normal_z, area —
        each a (num_points, 1) numpy array, matching the original API.
        """
        n_tri = self._mesh.n_cells
        probs = self._areas / self._total_area

        # Sample triangles proportional to area
        tri_ids = np.random.choice(n_tri, size=num_points, replace=True, p=probs)

        # Get triangle vertices
        faces = self._mesh.faces.reshape(-1, 4)[:, 1:]  # (n_tri, 3) vertex indices
        pts = np.asarray(self._mesh.points)              # (n_pts, 3)

        v0 = pts[faces[tri_ids, 0]]
        v1 = pts[faces[tri_ids, 1]]
        v2 = pts[faces[tri_ids, 2]]

        # Uniform random barycentric sampling (Osada et al.)
        r1 = np.random.rand(num_points, 1)
        r2 = np.random.rand(num_points, 1)
        sqrt_r1 = np.sqrt(r1)
        sampled = (1 - sqrt_r1) * v0 + sqrt_r1 * (1 - r2) * v1 + sqrt_r1 * r2 * v2

        normals = self._normals[tri_ids]                 # (num_points, 3)
        areas   = self._areas[tri_ids, np.newaxis]       # (num_points, 1)

        return {
            "x":        sampled[:, 0:1],
            "y":        sampled[:, 1:2],
            "z":        sampled[:, 2:3],
            "normal_x": normals[:, 0:1],
            "normal_y": normals[:, 1:2],
            "normal_z": normals[:, 2:3],
            "area":     areas,
        }

from dataloader import PartitionedGraph


def convert_to_triangular_mesh(
    polydata, write=False, output_filename="surface_mesh_triangular.vtu"
):
    """Converts a vtkPolyData object to a triangular mesh."""
    tet_filter = vtk.vtkDataSetTriangleFilter()
    tet_filter.SetInputData(polydata)
    tet_filter.Update()

    tet_mesh = pv.wrap(tet_filter.GetOutput())

    if write:
        tet_mesh.save(output_filename)

    return tet_mesh


def extract_surface_triangles(tet_mesh):
    """Extracts the surface triangles from a triangular mesh."""
    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(tet_mesh)
    surface_filter.Update()

    surface_mesh = pv.wrap(surface_filter.GetOutput())
    triangle_indices = []
    faces = surface_mesh.faces.reshape((-1, 4))
    for face in faces:
        if face[0] == 3:
            triangle_indices.extend([face[1], face[2], face[3]])
        else:
            raise ValueError("Face is not a triangle")

    return triangle_indices


def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i) for i in range(num_points)]
    return vertices


def add_edge_features(graph: pyg.data.Data) -> pyg.data.Data:
    """
    Add relative displacement and displacement norm as edge features to the graph.
    The calculations are done using the 'pos' attribute in the
    node data of each graph. The resulting edge features are stored in the 'x' attribute
    in the edge data of each graph.

    This method will modify the graph in-place.

    Returns
    -------
    pyg.data.Data
        Graph with updated edge features.
    """

    pos = graph.coordinates
    row, col = graph.edge_index

    disp = pos[row] - pos[col]
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    graph.edge_attr = torch.cat((disp, disp_norm), dim=-1)

    return graph


# Define this function outside of any local scope so it can be pickled
def run_task(params):
    """Wrapper function to unpack arguments for process_run."""
    return process_run(*params)


def process_partition(graph, num_partitions, halo_hops):
    """
    Helper function to partition a single graph and include node and edge features.
    """
    # Perform the partitioning
    return PartitionedGraph(graph, num_partitions, halo_hops)


def process_run(
    run_path, point_list, node_degree, num_partitions, halo_hops, save_point_cloud=False
):
    """Process a single run directory to generate a multi-level graph and apply partitioning."""
    run_id = os.path.basename(run_path).split("_")[-1]

    stl_file = os.path.join(run_path, f"drivaer_{run_id}_single_solid.stl")
    vtp_file = os.path.join(run_path, f"boundary_{run_id}.vtp")

    # Path to save the list of partitions
    partition_file_path = to_absolute_path(f"partitions/graph_partitions_{run_id}.bin")

    if os.path.exists(partition_file_path):
        print(f"Partitions for run {run_id} already exist. Skipping...")
        return

    if not os.path.exists(stl_file) or not os.path.exists(vtp_file):
        print(f"Warning: Missing files for run {run_id}. Skipping...")
        return

    try:
        # Load the STL and VTP files
        obj = Tessellation.from_stl(stl_file, airtight=False)
        surface_mesh = read_vtp(vtp_file)
        surface_mesh = convert_to_triangular_mesh(surface_mesh)
        surface_vertices = fetch_mesh_vertices(surface_mesh)
        surface_mesh = surface_mesh.cell_data_to_point_data()
        node_attributes = surface_mesh.point_data
        pressure_ref = node_attributes["pMeanTrim"]
        shear_stress_ref = node_attributes["wallShearStressMeanTrim"]

        # Sort the list of points in ascending order
        sorted_points = sorted(point_list)

        # Initialize arrays to store all points, normals, and areas
        all_points = np.empty((0, 3))
        all_normals = np.empty((0, 3))
        all_areas = np.empty((0, 1))
        edge_sources = []
        edge_destinations = []

        # Precompute the nearest neighbors for surface vertices
        nbrs_surface = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            surface_vertices
        )

        for num_points in sorted_points:
            # Sample the boundary points for the current level
            boundary = obj.sample_boundary(num_points)
            points = np.concatenate(
                [boundary["x"], boundary["y"], boundary["z"]], axis=1
            )
            normals = np.concatenate(
                [boundary["normal_x"], boundary["normal_y"], boundary["normal_z"]],
                axis=1,
            )
            area = boundary["area"]

            # Concatenate new points with the previous ones
            all_points = np.vstack([all_points, points])
            all_normals = np.vstack([all_normals, normals])
            all_areas = np.vstack([all_areas, area])

            # Construct edges for the combined point cloud at this level
            nbrs_points = NearestNeighbors(
                n_neighbors=node_degree + 1, algorithm="ball_tree"
            ).fit(all_points)
            _, indices_within = nbrs_points.kneighbors(all_points)
            src_within = [i for i in range(len(all_points)) for _ in range(node_degree)]
            dst_within = indices_within[:, 1:].flatten()

            # Add the within-level edges
            edge_sources.extend(src_within)
            edge_destinations.extend(dst_within)

        # Now, compute pressure and shear stress for the final combined point cloud
        _, indices = nbrs_surface.kneighbors(all_points)
        indices = indices.flatten()

        pressure = pressure_ref[indices]
        shear_stress = shear_stress_ref[indices]

    except Exception as e:
        print(f"Error processing run {run_id}: {e}. Skipping this run...")
        return

    try:
        # Create the final graph with multi-level edges
        edge_index = torch.stack(
            [
                torch.tensor(edge_sources, dtype=torch.long),
                torch.tensor(edge_destinations, dtype=torch.long),
            ],
            dim=0,
        )

        # Create a bidirectional graph object.
        edge_index = pyg.utils.coalesce(edge_index)
        edge_index = pyg.utils.to_undirected(edge_index)
        edge_index, _ = pyg.utils.add_self_loops(edge_index)

        graph = pyg.data.Data(
            edge_index=edge_index,
            coordinates=torch.tensor(all_points, dtype=torch.float32),
            normals=torch.tensor(all_normals, dtype=torch.float32),
            area=torch.tensor(all_areas, dtype=torch.float32),
            pressure=torch.tensor(pressure, dtype=torch.float32).unsqueeze(-1),
            shear_stress=torch.tensor(shear_stress, dtype=torch.float32),
        )

        graph = add_edge_features(graph)

        # PyG ClusterData uses `x` attribute of the source graph to set the number of nodes in each partition.
        # This is required to make ClusterData indexing work properly. The real value of `x` will
        # be set in a trainer, so set `x` to a NaN tensor to make sure it is not used.
        graph.x = torch.full((graph.coordinates.shape[0], 1), float("nan"))

        # Partition the graph
        partitioned_graphs = process_partition(graph, num_partitions, halo_hops)

        # Save the partitions
        os.makedirs(os.path.dirname(partition_file_path), exist_ok=True)
        torch.save(partitioned_graphs, partition_file_path)

        if save_point_cloud:
            parts = []
            for part in partitioned_graphs:
                point_cloud = pv.PolyData(part.coordinates.numpy())
                point_cloud["coordinates"] = part.coordinates.numpy()
                point_cloud["normals"] = part.normals.numpy()
                point_cloud["area"] = part.area.numpy()
                point_cloud["pressure"] = part.pressure.numpy()
                point_cloud["shear_stress"] = part.shear_stress.numpy()
                parts.append(point_cloud)

            multi_point_cloud = pv.MultiBlock(parts)
            for part_id in range(len(parts)):
                multi_point_cloud[part_id].name = part_id
            vtp_file_path = to_absolute_path(f"point_clouds/point_cloud_{run_id}.vtm")
            os.makedirs(os.path.dirname(vtp_file_path), exist_ok=True)
            multi_point_cloud.save(vtp_file_path)

    except Exception as e:
        print(
            f"Error while constructing graph or saving data for run {run_id}: {e}. Skipping this run..."
        )
        return


def process_all_runs(
    base_path,
    num_points,
    node_degree,
    num_partitions,
    halo_hops,
    num_workers=16,
    save_point_cloud=False,
):
    """Process all runs in the base directory in parallel."""

    run_dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_path, d))
    ]

    tasks = [
        (run_dir, num_points, node_degree, num_partitions, halo_hops, save_point_cloud)
        for run_dir in run_dirs
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in tqdm(
            pool.map(run_task, tasks),
            total=len(tasks),
            desc="Processing Runs",
            unit="run",
        ):
            pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    process_all_runs(
        base_path=to_absolute_path(cfg.data_path),
        num_points=cfg.num_nodes,
        node_degree=cfg.node_degree,
        num_partitions=cfg.num_partitions,
        halo_hops=cfg.num_message_passing_layers,
        num_workers=cfg.num_preprocess_workers,
        save_point_cloud=cfg.save_point_clouds,
    )


if __name__ == "__main__":
    main()