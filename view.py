import open3d as o3d
import sys

def view_ply_mesh(file_path):
    """
    Read and visualize a .ply mesh file
    """
    # Read the mesh file
    print(f"Loading mesh from {file_path}...")
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals()
    
    # Print mesh information
    print(f"Mesh contains {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Visualize the mesh
    print("Visualizing mesh. Press 'q' to exit the viewer.")
    o3d.visualization.draw_geometries([mesh], 
                                     zoom=0.6,
                                     window_name="PLY Mesh Viewer")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        view_ply_mesh(file_path)
    else:
        print("Please provide the path to a .ply file:")
        print("Usage: python view_ply.py path/to/mesh.ply")