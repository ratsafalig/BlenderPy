import sys
import open3d as o3d
from fbxloader import FBXLoader

def view_fbx_geometry(file_path):
    print(f"尝试使用 fbxloader 读取: {file_path}")
    try:
        loader = FBXLoader(file_path)
        # 将加载的数据转换为 Trimesh 或 Open3D 格式
        # 这里我们手动提取顶点和面来通过 Open3D 展示
        vertices = loader.vertices
        faces = loader.indices

        if vertices is None or len(vertices) == 0:
            print("错误: 未读取到顶点数据。")
            return

        # 创建 Open3D 网格
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces.reshape(-1, 3))

        mesh.compute_vertex_normals()

        print("读取成功！正在打开窗口...")
        o3d.visualization.draw_geometries([mesh], window_name="FBX Loader Viewer")

    except Exception as e:
        print(f"读取失败: {e}")

if __name__ == "__main__":
    # 替换为你的文件路径
    file_path = "MBU_S3_HB_Bridges.fbx"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    view_fbx_geometry(file_path)