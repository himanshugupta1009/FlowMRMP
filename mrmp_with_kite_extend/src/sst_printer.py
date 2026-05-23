# sst_printer.py
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import Optional, Iterable, Tuple

# If you use these types in hints:
# from Environments import SquareEnvObsShape

def _xy2(a: np.ndarray) -> np.ndarray:
    """Return x,y columns for an array of states (N,d)."""
    return a[:, :2]

def _edge_segments_from_polyline(parent_xy: np.ndarray, polyline_xy: np.ndarray) -> np.ndarray:
    """
    Build consecutive line segments for a single edge.
    Input:
      parent_xy: shape (2,)
      polyline_xy: shape (K,2) of points along the edge (does not include parent)
    Returns:
      (K, 2, 2) if K >= 1 else empty
    """
    if polyline_xy.shape[0] == 0:
        return np.empty((0, 2, 2), dtype=float)
    pts = np.vstack([parent_xy[None, :], polyline_xy])  # (K+1, 2)
    return np.stack([pts[:-1], pts[1:]], axis=1)        # (K, 2, 2)

def _collect_all_tree_segments(states: np.ndarray,
                               parents: np.ndarray,
                               path_from_parent: np.ndarray,
                               sub_path_length: np.ndarray,
                               count: int,
                               max_edges: Optional[int] = None) -> np.ndarray:
    """
    Convert the whole SST tree into a single (E,2,2) array of segments.
    Each node contributes segments between its parent and the polyline stored for that edge.
    """
    segs = []
    xy = states[:count, :2]
    for i in range(1, count):
        p = int(parents[i])
        if p < 0:
            continue
        k = int(sub_path_length[i])
        if k < 0:
            continue
        edge_xy = path_from_parent[i, :k, :2]
        segs.append(_edge_segments_from_polyline(xy[p], edge_xy))
        # Optional: if no samples were stored, you could fall back to straight segment:
        # if k == 0:
        #     segs.append(np.array([[xy[p], xy[i]]]))
    if not segs:
        return np.empty((0, 2, 2), dtype=float)
    all_segs = np.concatenate(segs, axis=0)
    if max_edges is not None and len(all_segs) > max_edges:
        idx = np.linspace(0, len(all_segs) - 1, max_edges).astype(int)
        all_segs = all_segs[idx]
    return all_segs

def _backtrack_indices(parents: np.ndarray, goal_idx: int) -> np.ndarray:
    """Return indices from root→goal by following `parents` from goal."""
    path = []
    i = int(goal_idx)
    while i != -1:
        path.append(i)
        i = int(parents[i])
    return np.array(path[::-1], dtype=int)

class SSTPrinter:
    """
    Array-native visualizer for your SST matrices.
    Mirrors the visual grammar of your RRTPrinter:
      - edges follow stored rollouts: [parent_state] + path_from_parent
      - final solution path highlighted
      - obstacles/start/goal drawn the same way
    Adds SST extras:
      - active vs inactive node styling
      - witness points & optional prune-radius circles
    """
    def __init__(self, env, sst, goal_path_indices: Optional[np.ndarray] = None):
        """
        env: same env object your RRTPrinter used (for size/obstacles)
        sst: your SST instance (to access _node_matrix / _witness_matrix, start/goal, etc.)
        goal_path_indices: optional array of node indices in the final path (root→goal).
                           If None and sst.path_found, we compute it from sst.goal_node_id.
        """
        self.env = env
        self.sst = sst

        self.nodes = sst._node_matrix           # SSTNodeMatrix
        self.witnesses = sst._witness_matrix    # SSTWitnessMatrix

        self.start = sst.start
        self.goal = sst.goal
        self.goal_radius = sst.goal_radius
        self.agent = sst.agent

        # Path nodes (indices into node arrays), root→goal
        if goal_path_indices is not None:
            self.path_indices = np.asarray(goal_path_indices, dtype=int)
        elif getattr(sst, "path_found", False) and sst.goal_node_id is not None:
            self.path_indices = _backtrack_indices(self.nodes.parent, int(sst.goal_node_id))
        else:
            self.path_indices = np.array([], dtype=int)

    # ------- obstacle helpers (copied interface) -------
    @staticmethod
    def get_obstacle_patch(ob, obs_buffer_value=0., color='grey'):
        from Environments import SquareEnvObsShape
        if ob.shape is SquareEnvObsShape.CIRCLE:
            return plt.Circle((ob.x, ob.y), ob.r + obs_buffer_value, color=color)
        elif ob.shape is SquareEnvObsShape.RECTANGLE:
            w = ob.w + obs_buffer_value
            h = ob.h + obs_buffer_value
            xy = (ob.x - w/2., ob.y - h/2.)
            return plt.Rectangle(xy, w, h, edgecolor='none', facecolor=color)

    @staticmethod
    def print_obs(ax, obs, obs_buffer_value=0.):
        plots = []
        for ob in obs:
            if obs_buffer_value > 0:
                ax.add_patch(SSTPrinter.get_obstacle_patch(ob, obs_buffer_value, color='xkcd:pale peach'))
            patch = ax.add_patch(SSTPrinter.get_obstacle_patch(ob, color='grey'))
            anno = ax.annotate("O", (ob.x, ob.y), color='black', ha='center', va='center')
            plots.append((patch, anno))
        return plots

    @staticmethod
    def print_goal(ax, goal, goal_radius, agent_id=""):
        patch = ax.add_patch(plt.Circle((goal[0], goal[1]), goal_radius, color='xkcd:light pastel green', label='Goal'))
        anno = ax.annotate(agent_id, (goal[0], goal[1]), xytext=(5,5), textcoords='offset points',
                           ha='center', va='bottom') if agent_id else None
        return (patch, anno)

    @staticmethod
    def print_start(ax, start, agent, pcol='black', agent_id=""):
        patch = ax.add_patch(plt.Circle((start[0], start[1]), agent.radius, color=pcol))
        anno = ax.annotate(agent_id, (start[0], start[1]), ha='center', va='center') if agent_id else None
        return (patch, anno)

    # ------- main static image -------
    def print_sst(self,
                  filename: str,
                  show_tree: bool = True,
                  show_path: bool = True,
                  show_active: bool = True,
                  show_inactive: bool = True,
                  show_witness: bool = True,
                  show_prune_circles: bool = False,
                  max_edges: Optional[int] = None):
        """
        Render SST tree & path to an image, mirroring your RRTPrinter aesthetics.
        """
        nm = self.nodes
        wm = self.witnesses
        N = nm.count

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_aspect('equal', adjustable='box')

        # obstacles
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)

        # goal/start
        self.print_goal(ax, self.goal, self.goal_radius)
        self.print_start(ax, self.start, self.agent, pcol='black')

        # tree edges (polyline) with a single LineCollection
        if show_tree and N > 1:
            segments = _collect_all_tree_segments(
                nm.state, nm.parent, nm.path_from_parent, nm.sub_path_length, N, max_edges=max_edges
            )
            if len(segments):
                lc = LineCollection(segments, linewidths=0.8, alpha=0.6, color='y')
                ax.add_collection(lc)

        # nodes: active vs inactive
        xy = nm.state[:N, :2]
        if hasattr(nm, "active") and nm.active is not None:
            active_mask = (nm.active[:N] == 1)
        else:
            active_mask = np.ones(N, dtype=bool)

        if show_inactive and np.any(~active_mask):
            ax.scatter(xy[~active_mask, 0], xy[~active_mask, 1], s=6, alpha=0.35, color='y')
        if show_active and np.any(active_mask):
            ax.scatter(xy[active_mask, 0], xy[active_mask, 1], s=10, alpha=0.9, color='orange')

        # solution path: redraw those edges thicker and nodes on top
        if show_path and self.path_indices.size > 1:
            path_pairs = []
            for i in self.path_indices[1:]:  # skip root
                p = int(nm.parent[i])
                if p < 0:
                    continue
                k = int(nm.sub_path_length[i])
                seg_xy = nm.path_from_parent[i, :k, :2]
                # Build consecutive pairs including the jump from parent
                segs = _edge_segments_from_polyline(xy[p], seg_xy)
                if segs.size:
                    path_pairs.append(segs)
            if path_pairs:
                path_lc = LineCollection(np.concatenate(path_pairs, axis=0), linewidths=2.3, alpha=0.95, color='c')
                ax.add_collection(path_lc)
            # End marker on goal node index if available
            goal_idx = int(self.sst.goal_node_id) if getattr(self.sst, "goal_node_id", None) is not None else self.path_indices[-1]
            ax.scatter(xy[goal_idx,0], xy[goal_idx,1], s=40, edgecolor='k', facecolor='none')

        # witnesses + representatives
        if show_witness and wm.count > 0:
            wxy = wm.state[:wm.count, :2]
            ax.scatter(wxy[:,0], wxy[:,1], s=18, marker='x', color='k', alpha=0.8)
            reps = wm.rep_index[:wm.count]
            valid = reps >= 0
            if np.any(valid):
                ax.scatter(xy[reps[valid],0], xy[reps[valid],1], s=28, facecolors='none',
                           edgecolors='k', linewidths=1.3)
            if show_prune_circles and hasattr(self.sst, "prune_radius") and self.sst.prune_radius is not None:
                for (wx, wy) in wxy:
                    ax.add_patch(plt.Circle((wx, wy), self.sst.prune_radius, fill=False, alpha=0.15, color='k'))

        fig.savefig(filename)
        plt.close(fig)

    # ------- step-wise animation like print_rrt_step_ani -------
    def print_sst_step_ani(self, filename: str, animation_speed: int = 2,
                           show_tree=True, show_path=True):
        """
        Simple insertion-order animation: at frame n, draw node n's edge.
        If it's on the final path, also draw it with path styling.
        """
        import matplotlib.animation as animation

        nm = self.nodes
        N = nm.count

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.env.size[0])
        ax.set_ylim(0, self.env.size[1])
        ax.set_aspect('equal', adjustable='box')

        # static stuff
        self.print_obs(ax, self.env.obstacles, self.env.obstacle_buffer)
        self.print_goal(ax, self.goal, self.goal_radius)
        self.print_start(ax, self.start, self.agent, pcol='black')

        path_set = set(self.path_indices.tolist())
        xy = nm.state[:N, :2]

        def animate(n):
            plots = []
            if n == 0:
                return plots
            i = n
            if i >= N:
                i = N - 1
            p = int(nm.parent[i])
            if p >= 0:
                k = int(nm.sub_path_length[i])
                seg_xy = nm.path_from_parent[i, :k, :2]
                xs = [xy[p,0]] + seg_xy[:,0].tolist()
                ys = [xy[p,1]] + seg_xy[:,1].tolist()
                if show_tree:
                    plots += ax.plot(xs, ys, color='y', linestyle='-')
                if show_path and i in path_set:
                    plots += ax.plot(xs, ys, color='c', linewidth=2.0)
                # node dot
                plots += ax.plot(xy[i,0], xy[i,1], color='orange', marker='.', markersize=3)
            return plots

        ani = animation.FuncAnimation(fig, animate, interval=animation_speed, frames=N, blit=True)
        ani.save(filename=filename, writer="pillow")
        plt.close(fig)



"""

from sst_printer import SSTPrinter

printer = SSTPrinter(env, sst)
printer.print_sst("media/sst_graph.png",
                  show_tree=True, show_path=True,
                  show_active=True, show_inactive=False,
                  show_witness=True, show_prune_circles=False)

# optional animation
printer.print_sst_step_ani("media/sst_build.gif", animation_speed=2)


"""