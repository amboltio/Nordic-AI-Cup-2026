import numpy as np

def compute_visibility(x, y, direction, cone_angle, vision_radius, edges, epsilon=1e-3):
    """
    Compute visibility polygon by raycasting to obstacle corners.

    Args:
        x, y (float): Agent position.
        direction (float): Agent direction.
        cone_angle (float): Vision cone angle.
        vision_radius (float): Vision distance.
        edges (list of ((x1, y1), (x2, y2)): Obstacle edges.
        epsilon (float): Offset for raycasting.
    
    Returns:
        polygon_points (list) : (x,y) points forming the visibility polygon.
        hit_edges (list of ((x1, y1), (x2, y2)): Edges that block vision.
    """

    polygon_points = [] # Array to hold the points of the visibility polygon
    hit_edges = [] # Array to hold the edges that block vision

    # Collect corners from edges
    corners = np.array([pt for edge in edges for pt in edge]) # shape (N,2)

    # Vectors from agent to corners
    dx = corners[:, 0] - x # All x coords
    dy = corners[:, 1] - y # All y coords

    # Filter: within vision_radius
    dist_sq = dx**2 + dy**2
    mask = dist_sq <= vision_radius**2 # True if within vision_radius. Shape (N,)
    dx, dy = dx[mask], dy[mask] # Remove points too far away

    rays = [] # Store rays to cast
    if dx.size > 0: # If there are any corners in range
        base_angles = np.arctan2(dy, dx) # Angles to all corners
        offsets = np.array([-epsilon, 0, epsilon]) # Add offset to see "around" corners
        all_angles = base_angles[:, None] + offsets  # Combine the base angles with offsets (shape (N,3). One at the corner and one on each side)
        angle_rel = (all_angles - direction + np.pi) % (2*np.pi) - np.pi # wrap relative angles to [-pi, pi]
        mask = np.abs(angle_rel) <= cone_angle/2 # Keep only angles within vision cone
        rays.extend(all_angles[mask].tolist())

    # Add 5 rays to ensure the vision cone is constructed even if no/few corners are present
    rays.extend([
        direction - cone_angle/2, # far left
        direction + cone_angle/2, # far right
        direction - cone_angle/4, # mid left
        direction + cone_angle/4,# mid right
        direction # mid
    ])
    rays = np.unique(rays)  # sort and remove duplicates

    # Vectorize ray/edge intersections
    rays_dx = np.cos(rays) # x components of rays directions unit vectors
    rays_dy = np.sin(rays) # y components of rays directions unit vectors

    edges_arr = np.array(edges, dtype = float)  # Array of edges with Start and End points each with an x and y. shape (Edges,2,2)
    if edges_arr.size == 0:
        # No edges: polygon is just rays extended to vision_radius, no hit edges
        px = x + rays_dx * vision_radius
        py = y + rays_dy * vision_radius
        polygon_points = list(zip(px, py))
        hit_edges = []
        return polygon_points, hit_edges  # <-- safe early return

    x1, y1 = edges_arr[:, 0, 0], edges_arr[:, 0, 1] # Start points of edges
    x2, y2 = edges_arr[:, 1, 0], edges_arr[:, 1, 1] # End points of edges

    vx = x2 - x1 # Edge vector x component
    vy = y2 - y1 # Edge vector y component

    # Broadcast to shape (num_rays, num_edges)
    det = -rays_dx[:, None] * vy[None, :] + rays_dy[:, None] * vx[None, :] # Determinant (if 0, ray and edge are parallel)
    mask_det = np.abs(det) >= 1e-8 # Mask to avoid division by zero

    # Avoid division by creating a safe version of det
    det_safe = np.where(mask_det, det, 1.0)  # Replace small values with 1.0


    t = np.where(mask_det,
                 (-vy[None, :] * (x1[None, :] - x) + vx[None, :] * (y1[None, :] - y)) / det_safe,
                 np.inf) # Ray parameter (distance along ray where intersection occurs)

    u = np.where(mask_det,
                 (-rays_dy[:, None] * (x1[None, :] - x) + rays_dx[:, None] * (y1[None, :] - y)) / det_safe,
                 np.inf) # Edge parameter (0=start of edge, 1=end of edge, between 0 and 1 means intersection occurs on the edge segment)

    valid = (t >= 0) & (u >= 0) & (u <= 1) # Valid intersections only (in front of ray and on edge segment)

    # Replace invalid with inf
    t[~valid] = np.inf

    # Find closest intersection per ray
    min_idx = np.argmin(t, axis=1) # Index of closest edge per ray
    min_t = t[np.arange(len(rays)), min_idx] # Closest t per ray

    # Clip by vision_radius (Avoid going further than vision_radius if no intersection is found)
    min_t = np.minimum(min_t, vision_radius)

    # Compute intersection points
    px = x + rays_dx * min_t # x coords of intersection points
    py = y + rays_dy * min_t # y coords of intersection points
    polygon_points = list(zip(px, py)) # Convert to list of (x,y) points

    # Collect hit edges
    hit_mask = min_t < vision_radius # If smallest t is less than vision_radius, we hit an edge
    hit_edges = [tuple(map(tuple, edges_arr[j])) for j in min_idx[hit_mask]] # Convert to list of edges that were hit

    def relative_angle(p, x, y, direction):
        angle = np.arctan2(p[1]-y, p[0]-x) - direction
        return (angle + np.pi) % (2*np.pi) - np.pi

    # Sort points by relative angle
    polygon_points.sort(key=lambda p: relative_angle(p, x, y, direction))

    return polygon_points, hit_edges
