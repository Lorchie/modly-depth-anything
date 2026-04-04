import os
import cv2
import numpy as np
import open3d as o3d
import torch
from modly_sdk import BaseGenerator
from depth_anything import DepthAnything

class DepthAnythingGenerator(BaseGenerator):

    def load_model(self):
        model_type = self.params.get("model_type", "vitl")
        use_cuda = self.params.get("use_cuda", "auto")

        if use_cuda == "true":
            self.device = "cuda"
        elif use_cuda == "false":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = DepthAnything.from_pretrained(model_type).to(self.device)

    def generate(self, image_path: str) -> str:
        node = self.params.get("nodeId")

        if node == "image_to_depth":
            return self.node_image_to_depth(image_path)

        if node == "depth_to_pointcloud":
            return self.node_depth_to_pointcloud(image_path)

        if node == "pointcloud_to_mesh":
            return self.node_pointcloud_to_mesh(image_path)

        raise ValueError(f"Unknown node: {node}")

    # ───────────────────────────────────────────────
    # 1) IMAGE → DEPTH
    # ───────────────────────────────────────────────
    def node_image_to_depth(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = self.model.infer_image(img_rgb)

        depth_norm = (255 * (depth - depth.min()) / (depth.max() - depth.min())).astype(np.uint8)

        output_path = os.path.join(self.workspace_dir, "depth.png")
        cv2.imwrite(output_path, depth_norm)

        return output_path

    # ───────────────────────────────────────────────
    # 2) DEPTH → POINT CLOUD
    # ───────────────────────────────────────────────
    def node_depth_to_pointcloud(self, depth_path):
        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        h, w = depth.shape

        fx = fy = 500
        cx = w / 2
        cy = h / 2

        points = []
        for y in range(h):
            for x in range(w):
                z = depth[y, x] / 255.0
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                points.append([X, -Y, z])

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)

        output_path = os.path.join(self.workspace_dir, "pointcloud.ply")
        o3d.io.write_point_cloud(output_path, pc)

        return output_path

    # ───────────────────────────────────────────────
    # 3) POINT CLOUD → MESH
    # ───────────────────────────────────────────────
    def node_pointcloud_to_mesh(self, pc_path):
        pc = o3d.io.read_point_cloud(pc_path)

        pc.estimate_normals()
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=8)

        mesh = mesh.simplify_quadric_decimation(50000)

        output_path = os.path.join(self.workspace_dir, "mesh.glb")
        o3d.io.write_triangle_mesh(output_path, mesh)

        return output_path