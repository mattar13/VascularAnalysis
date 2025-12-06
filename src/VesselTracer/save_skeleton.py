import json
import numpy as np
import networkx as nx
from scipy import ndimage
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple


def paths_to_graph(paths: Dict[int, Dict[str, Any]], 
                   pixel_sizes: Optional[Tuple[float, float, float]] = None,
                   binary_volume: Optional[np.ndarray] = None,
                   coordinate_order: str = 'zyx') -> Dict[str, Any]:
    """
    Convert vessel paths dictionary to a graph representation with nodes and edges.
    
    The output format is:
    {
        "nodes": [{"id": 0, "x": 50, "y": 100, "z": 10}, ...],
        "edges": [{"source": 0, "target": 1, "length": 5.2, "radius": 2.0, "path_id": 1}, ...]
    }
    
    Args:
        paths: Dictionary of paths, where each path has:
            - 'coordinates': numpy array of shape (n_points, 3) with [z, y, x] coordinates
            - 'length': number of points in path
        pixel_sizes: Optional tuple of (z, y, x) pixel sizes in microns for unit conversion
        binary_volume: Optional binary volume for radius estimation using distance transform
        coordinate_order: Order of coordinates in paths ('zyx' or 'xyz')
        
    Returns:
        Dictionary with 'nodes' and 'edges' lists
    """
    if paths is None or len(paths) == 0:
        return {"nodes": [], "edges": []}
    
    # Set default pixel sizes if not provided
    if pixel_sizes is None:
        z_scale, y_scale, x_scale = 1.0, 1.0, 1.0
    else:
        z_scale, y_scale, x_scale = pixel_sizes
    
    # Compute distance transform for radius estimation if binary volume provided
    distance_transform = None
    if binary_volume is not None:
        distance_transform = ndimage.distance_transform_edt(binary_volume)
    
    # Track unique nodes and their IDs
    nodes = []
    edges = []
    node_id_counter = 0
    
    # Process each path
    for path_id, path_info in paths.items():
        coords = path_info['coordinates']
        
        if len(coords) < 2:
            # Single point path - just add as node with no edges
            if coordinate_order == 'zyx':
                z, y, x = coords[0]
            else:
                x, y, z = coords[0]
            
            nodes.append({
                "id": node_id_counter,
                "x": float(x * x_scale),
                "y": float(y * y_scale),
                "z": float(z * z_scale),
                "path_id": int(path_id)
            })
            node_id_counter += 1
            continue
        
        # Track nodes for this path
        path_node_ids = []
        
        for i, coord in enumerate(coords):
            if coordinate_order == 'zyx':
                z, y, x = coord
            else:
                x, y, z = coord
            
            # Create node
            node = {
                "id": node_id_counter,
                "x": float(x * x_scale),
                "y": float(y * y_scale),
                "z": float(z * z_scale),
                "path_id": int(path_id)
            }
            nodes.append(node)
            path_node_ids.append(node_id_counter)
            node_id_counter += 1
        
        # Create edges between consecutive nodes
        for i in range(len(path_node_ids) - 1):
            source_id = path_node_ids[i]
            target_id = path_node_ids[i + 1]
            
            # Get coordinates for edge endpoints
            if coordinate_order == 'zyx':
                z1, y1, x1 = coords[i]
                z2, y2, x2 = coords[i + 1]
            else:
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[i + 1]
            
            # Calculate edge length in scaled coordinates
            dx = (x2 - x1) * x_scale
            dy = (y2 - y1) * y_scale
            dz = (z2 - z1) * z_scale
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Estimate radius from distance transform if available
            radius = 1.0  # Default radius
            if distance_transform is not None:
                try:
                    # Get midpoint coordinates (in original pixel space)
                    mid_z = int((z1 + z2) / 2)
                    mid_y = int((y1 + y2) / 2)
                    mid_x = int((x1 + x2) / 2)
                    
                    # Clamp to volume bounds
                    mid_z = max(0, min(mid_z, distance_transform.shape[0] - 1))
                    mid_y = max(0, min(mid_y, distance_transform.shape[1] - 1))
                    mid_x = max(0, min(mid_x, distance_transform.shape[2] - 1))
                    
                    radius = float(distance_transform[mid_z, mid_y, mid_x])
                except (IndexError, ValueError):
                    radius = 1.0
            
            edge = {
                "source": source_id,
                "target": target_id,
                "length": float(length),
                "radius": radius,
                "path_id": int(path_id)
            }
            edges.append(edge)
    
    return {"nodes": nodes, "edges": edges}


def export_paths_to_json(paths: Dict[int, Dict[str, Any]],
                         output_path: Union[str, Path],
                         pixel_sizes: Optional[Tuple[float, float, float]] = None,
                         binary_volume: Optional[np.ndarray] = None,
                         coordinate_order: str = 'zyx',
                         indent: int = 2) -> None:
    """
    Export vessel paths to a JSON file in graph format.
    
    Args:
        paths: Dictionary of paths from vessel tracing
        output_path: Path to save the JSON file
        pixel_sizes: Optional tuple of (z, y, x) pixel sizes in microns
        binary_volume: Optional binary volume for radius estimation
        coordinate_order: Order of coordinates ('zyx' or 'xyz')
        indent: JSON indentation level (use None for compact output)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    graph_data = paths_to_graph(
        paths=paths,
        pixel_sizes=pixel_sizes,
        binary_volume=binary_volume,
        coordinate_order=coordinate_order
    )
    
    with open(output_path, 'w') as f:
        json.dump(graph_data, f, indent=indent)
    
    print(f"Exported {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges to {output_path}")


def paths_to_networkx(paths: Dict[int, Dict[str, Any]],
                      pixel_sizes: Optional[Tuple[float, float, float]] = None,
                      binary_volume: Optional[np.ndarray] = None,
                      coordinate_order: str = 'zyx') -> nx.Graph:
    """
    Convert vessel paths to a NetworkX graph.
    
    Args:
        paths: Dictionary of paths from vessel tracing
        pixel_sizes: Optional tuple of (z, y, x) pixel sizes in microns
        binary_volume: Optional binary volume for radius estimation
        coordinate_order: Order of coordinates ('zyx' or 'xyz')
        
    Returns:
        NetworkX Graph with node positions and edge attributes
    """
    graph_data = paths_to_graph(
        paths=paths,
        pixel_sizes=pixel_sizes,
        binary_volume=binary_volume,
        coordinate_order=coordinate_order
    )
    
    G = nx.Graph()
    
    # Add nodes with attributes
    for node in graph_data['nodes']:
        G.add_node(
            node['id'],
            x=node['x'],
            y=node['y'],
            z=node['z'],
            path_id=node['path_id'],
            pos=(node['x'], node['y'], node['z'])
        )
    
    # Add edges with attributes
    for edge in graph_data['edges']:
        G.add_edge(
            edge['source'],
            edge['target'],
            length=edge['length'],
            radius=edge['radius'],
            path_id=edge['path_id']
        )
    
    return G


def export_paths_to_networkx_json(paths: Dict[int, Dict[str, Any]],
                                   output_path: Union[str, Path],
                                   pixel_sizes: Optional[Tuple[float, float, float]] = None,
                                   binary_volume: Optional[np.ndarray] = None,
                                   coordinate_order: str = 'zyx') -> None:
    """
    Export vessel paths to a NetworkX-compatible JSON format (node-link data).
    
    Args:
        paths: Dictionary of paths from vessel tracing
        output_path: Path to save the JSON file
        pixel_sizes: Optional tuple of (z, y, x) pixel sizes in microns
        binary_volume: Optional binary volume for radius estimation
        coordinate_order: Order of coordinates ('zyx' or 'xyz')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    G = paths_to_networkx(
        paths=paths,
        pixel_sizes=pixel_sizes,
        binary_volume=binary_volume,
        coordinate_order=coordinate_order
    )
    
    # Convert to node-link data format
    data = nx.node_link_data(G)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to {output_path}")
