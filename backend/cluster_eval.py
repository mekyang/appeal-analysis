import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.font_manager as fm

# 尝试设置中文显示 (根据系统不同可能需要调整)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ClusterEvaluator:
    """
    聚类效果评估工具箱 (Cluster Evaluation Toolkit)
    
    用于对 TaxClusteringEngine 的输出结果进行数学指标计算、可视化和语义分析。
    """

    def __init__(self, df: pd.DataFrame, embeddings: np.ndarray):
        """
        [Public] 初始化评估器。

        Args:
            df (pd.DataFrame): 包含 'Text', 'Cluster', 'Keywords' 列的结果表。
            embeddings (np.ndarray): 对应的 SBERT 原始向量 (或 UMAP 降维后的向量)。
                                     建议传入原始 SBERT 向量以获得更准确的语义距离。
        """
        self.df = df
        self.embeddings = embeddings
        
        # 预计算一些基础掩码
        self.valid_mask = self.df['Cluster'] != -1
        self.noise_mask = self.df['Cluster'] == -1
        self.n_clusters = len(self.df[self.valid_mask]['Cluster'].unique())

    def compute_metrics(self) -> dict:
        """
        [Public] 计算核心数学指标。
        
        Returns:
            dict: 包含噪音率、轮廓系数等指标的字典。
        """
        print("📊 [Metric] 正在计算数学指标...")
        
        # 1. 噪音比例
        total = len(self.df)
        noise_count = self.noise_mask.sum()
        noise_ratio = noise_count / total
        
        metrics = {
            "Total Samples": total,
            "Valid Clusters": self.n_clusters,
            "Noise Ratio": f"{noise_ratio:.2%}"
        }

        # 2. 轮廓系数 (Silhouette Score)
        # 注意：轮廓系数计算量大，且不能包含噪音点，至少要有2个簇
        if self.n_clusters > 1:
            valid_embeddings = self.embeddings[self.valid_mask]
            valid_labels = self.df[self.valid_mask]['Cluster']
            
            # 使用余弦距离计算
            score = silhouette_score(valid_embeddings, valid_labels, metric='cosine')
            metrics['Silhouette Score'] = round(score, 4)
            
            # Calinski-Harabasz Score (方差比标准) - 分数越高越好
            ch_score = calinski_harabasz_score(valid_embeddings, valid_labels)
            metrics['CH Score'] = round(ch_score, 2)
        else:
            metrics['Silhouette Score'] = "N/A (簇数量不足)"

        return metrics

    def plot_size_distribution(self, top_n: int = 20):
        """
        [Public] 绘制聚类大小分布图 (柱状图)。
        用于发现是否存在“巨型簇”或“长尾碎片”。
        """
        print("📊 [Plot] 正在绘制分布图...")
        plt.figure(figsize=(12, 6))
        
        # 统计每个簇的数量（不含噪音）
        counts = self.df[self.valid_mask]['Cluster'].value_counts().head(top_n)
        
        # 获取对应的关键词作为 X 轴标签
        cluster_labels = []
        for cid in counts.index:
            kw = self.df[self.df['Cluster'] == cid]['Keywords'].iloc[0]
            # 截取前两个关键词，避免图表太挤
            short_kw = ",".join(kw.split(',')[:2]) 
            cluster_labels.append(f"C{cid}\n{short_kw}")

        sns.barplot(x=cluster_labels, y=counts.values, palette="viridis")
        
        plt.title(f"Top {top_n} Largest Clusters Distribution")
        plt.xlabel("Cluster ID & Keywords")
        plt.ylabel("Number of Records")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_2d_scatter(self, output_path: str = None):
        """
        [Public] 绘制 2D 散点图可视化。
        
        Args:
            output_path (str): 如果提供路径，将保存图片。
        """
        print("🎨 [Plot] 正在降维并绘制 2D 散点图...")
        
        # 为了画图，我们需要将向量降到 2D
        # 注意：这里我们在该类内部重新跑一次 UMAP 2D，仅用于画图，不影响之前的聚类结果
        reducer_2d = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', random_state=42)
        embedding_2d = reducer_2d.fit_transform(self.embeddings)
        
        plt.figure(figsize=(14, 10))
        
        # 1. 画噪音 (灰色)
        if self.noise_mask.any():
            plt.scatter(embedding_2d[self.noise_mask, 0], 
                        embedding_2d[self.noise_mask, 1],
                        c='#E0E0E0', s=5, label='Noise', alpha=0.5)
            
        # 2. 画有效聚类
        # 使用 tab20 颜色板，区分度较高
        scatter = plt.scatter(embedding_2d[self.valid_mask, 0], 
                              embedding_2d[self.valid_mask, 1],
                              c=self.df[self.valid_mask]['Cluster'], 
                              cmap='tab20', s=8, alpha=0.8)
        
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('Tax Issues 2D Visualization')
        plt.xlabel('UMAP Dim 1')
        plt.ylabel('UMAP Dim 2')
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"   -> 图片已保存至: {output_path}")
        plt.show()

    def analyze_similarity(self):
        """
        [Public] 计算簇中心相似度热力图。
        帮助发现：是否有两个簇其实是在说同一件事（应该合并）？
        """
        if self.n_clusters < 2:
            print("❌ 簇数量不足，无法分析相似度。")
            return

        print("🔍 [Analysis] 正在分析簇间语义重叠度...")
        
        # 1. 计算每个簇的“质心” (Centroid) - 即该簇所有向量的平均值
        cluster_ids = sorted(self.df[self.valid_mask]['Cluster'].unique())
        centroids = []
        labels = []
        
        for cid in cluster_ids:
            # 获取该簇的所有向量
            indices = self.df[self.df['Cluster'] == cid].index
            cluster_vecs = self.embeddings[indices]
            centroid = np.mean(cluster_vecs, axis=0)
            centroids.append(centroid)
            
            # 获取标签用于画图
            kw = self.df[self.df['Cluster'] == cid]['Keywords'].iloc[0]
            labels.append(f"C{cid}: {kw.split(',')[0]}") # 只取第一个关键词

        # 2. 计算余弦相似度矩阵
        sim_matrix = cosine_similarity(centroids)
        
        # 3. 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, 
                    cmap="RdBu_r", center=0.5, annot=False)
        plt.title("Cluster Semantic Similarity Matrix (1.0 = Highly Similar)")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # 4. 自动给出建议
        # 找出相似度大于 0.85 的非对角线元素
        print("\n--- ⚠️ 合并建议 (Similarity > 0.85) ---")
        found = False
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                if sim_matrix[i][j] > 0.85:
                    print(f"建议检查: [{labels[i]}] <==> [{labels[j]}] (相似度: {sim_matrix[i][j]:.3f})")
                    found = True
        if not found:
            print("未发现明显的重叠簇，聚类区分度良好。")

    def run_full_report(self):
        """
        [Public] 一键运行所有体检项目
        """
        print("="*30)
        print("  CLUSTER EVALUATION REPORT  ")
        print("="*30)
        
        # 1. 指标
        metrics = self.compute_metrics()
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("-" * 30)
        
        # 2. 分布
        self.plot_size_distribution()
        
        # 3. 散点图
        self.plot_2d_scatter()
        
        # 4. 相似度
        self.analyze_similarity()