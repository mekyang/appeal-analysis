import pandas as pd
import numpy as np
import umap
import hdbscan
import jieba
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Union

class TaxClusteringEngine:
    """
    税务文本聚类分析引擎 (Tax Clustering Engine)
    
    基于 Sentence-BERT + UMAP + HDBSCAN 的现代文本挖掘流程。
    用于将非结构化的税务咨询文本转化为结构化的聚类话题。
    """

    def __init__(self, model_name: str = None):
        """
        [Public] 初始化分析引擎，优先从本地目录加载 SBERT 模型。

        Args:
            model_name (str): 本地模型路径或 HuggingFace 名称。若为 None，则默认使用
                              工作区下的本地模型 `models/BAAI/bge-large-zh-v1.5`。
        """
        # 优先使用本地模型目录，降低在线加载或 repo id 校验失败的风险
        default_local = os.path.join('models', 'BAAI', 'bge-large-zh-v1.5')

        # 如果用户指定了 model_name，检查它是否为本地路径
        if model_name:
            if os.path.exists(model_name):
                chosen_model = model_name
            else:
                # 尝试在 models/ 目录下查找
                candidate = os.path.join('models', model_name)
                if os.path.exists(candidate):
                    chosen_model = candidate
                else:
                    print(f"⚠️ 提供的模型路径 '{model_name}' 未找到，本地回退到 {default_local}")
                    chosen_model = default_local
        else:
            chosen_model = default_local

        print(f"🤖 [Init] 正在加载模型: {chosen_model}...")
        try:
            self.model = SentenceTransformer(chosen_model)
        except Exception as e:
            print(f"❌ 本地加载失败 ({e})，尝试使用在线模型 paraphrase-multilingual-MiniLM-L12-v2 加载...")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # --- 内部状态变量 (State) ---
        self._embeddings = None       # SBERT 原始向量
        self._umap_embeddings = None  # 降维后向量
        self._labels = None           # 聚类标签
        self.results_df = None        # 最终结果表
        
    def run_analysis(self, 
                     texts: List[str],
                     original: Optional[List[Union[int, str]]] = None, 
                     n_neighbors: int = 15, 
                     n_components: int = 5, 
                     min_cluster_size: int = 10,
                     keyword_top_n: int = 5,
                     progress_callback=None) -> pd.DataFrame:
        """
        [Public] 执行完整的分析流程：编码 -> 降维 -> 聚类 -> 关键词提取。

        Args:
            texts (List[str]): 已经预处理好的文本列表（List of strings）。
            original (List[Union[int, str]]): 原始数据的 ID 列表（如 Trace_ID）。
            n_neighbors (int): UMAP 参数。控制局部结构，越小越关注细节。建议 10-30。
            n_components (int): UMAP 参数。降维后的维度数，HDBSCAN 推荐 5-10。
            min_cluster_size (int): HDBSCAN 参数。最小成团数量，小于此数量视为噪音。
            keyword_top_n (int): 每个聚类提取多少个核心关键词。
            progress_callback (callable): 可选的进度回调函数，签名为 progress_callback(current, total, stage)

        Returns:
            pd.DataFrame: 包含原始文本、聚类ID、核心关键词的完整数据框。
        """
        if not texts:
            raise ValueError("输入文本列表不能为空！")
        
        if original is not None and len(texts) != len(original):
            raise ValueError(f"文本数量 ({len(texts)}) 与 原始数据ID 数量 ({len(original)}) 不一致！")

        print(f"🚀 开始分析 {len(texts)} 条数据...")

        # 1. 向量化
        self._encode_text(texts, progress_callback)

        # 2. 降维
        if progress_callback:
            progress_callback(35, 100, "📉 UMAP降维中...")
        self._reduce_dimensions(n_neighbors, n_components)

        # 3. 聚类
        if progress_callback:
            progress_callback(60, 100, "🔍 HDBSCAN聚类中...")
        self._perform_clustering(min_cluster_size)

        # 4. 结果整合与关键词提取
        if progress_callback:
            progress_callback(85, 100, "🏷️ 提取关键词中...")
        self._generate_final_report(texts, keyword_top_n, original)

        if progress_callback:
            progress_callback(100, 100, "✅ 分析完成！")
        
        print("✅ 分析流程结束。")
        return self.results_df

    def _encode_text(self, texts: List[str], progress_callback=None) -> None:
        """
        [Private] 使用 SBERT 将文本转化为高维语义向量。
        结果存储于 self._embeddings
        """
        print("🧠 [Step 1] 正在进行语义编码 (SBERT)...")
        
        # SBERT 编码 - 使用 show_progress_bar=False 避免与前端进度条冲突
        # 手动模拟进度
        batch_size = 32
        total_batches = (len(texts) + batch_size - 1) // batch_size
        embeddings_list = []
        
        for i, batch_start in enumerate(range(0, len(texts), batch_size)):
            batch_end = min(batch_start + batch_size, len(texts))
            batch = texts[batch_start:batch_end]
            
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings_list.append(batch_embeddings)
            
            if progress_callback:
                current = batch_end
                progress = int(5 + (current / len(texts)) * 25)  # 进度 5% -> 30%
                progress_callback(progress, 100, f"🧠 SBERT编码中 ({i+1}/{total_batches})...")
        
        self._embeddings = np.vstack(embeddings_list)
        
        if progress_callback:
            progress_callback(30, 100, "🧠 SBERT编码完成...")

    def _reduce_dimensions(self, n_neighbors: int, n_components: int) -> None:
        """
        [Private] 使用 UMAP 对向量进行降维，以便于密度聚类。
        结果存储于 self._umap_embeddings
        """
        print(f"📉 [Step 2] 正在降维 UMAP (neighbors={n_neighbors}, dim={n_components})...")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='cosine',  # 文本数据推荐使用余弦距离
            random_state=42   # 固定随机种子，保证结果可复现
        )
        self._umap_embeddings = reducer.fit_transform(self._embeddings)

    def _perform_clustering(self, min_cluster_size: int) -> None:
        """
        [Private] 使用 HDBSCAN 进行基于密度的聚类。
        结果存储于 self._labels
        """
        print(f"🔍 [Step 3] 正在聚类 HDBSCAN (min_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',        # 在低维空间使用欧氏距离即可
            cluster_selection_method='eom'
        )
        self._labels = clusterer.fit_predict(self._umap_embeddings)
        
        # 简单统计
        unique_labels = set(self._labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_ratio = list(self._labels).count(-1) / len(self._labels)
        print(f"   -> 发现 {n_clusters} 个聚类，噪音比例: {noise_ratio:.2%}")

    def _extract_cluster_keywords(self, df: pd.DataFrame, top_n: int) -> Dict[int, str]:
        """
        [Private] 基于 c-TF-IDF 思想提取每一类的关键词。
        
        Args:
            df (pd.DataFrame): 包含 'Cluster' 和 'Text' 列的临时表。
            top_n (int): 提取前 N 个词。
            
        Returns:
            Dict[int, str]: {聚类ID: "关键词1, 关键词2..."}
        """
        print("🏷️ [Step 4] 正在提取核心关键词...")
        
        # 1. 按聚类合并文本 (Document Aggregation)
        # 忽略噪音数据 (-1)
        docs_df = df[df['Cluster'] != -1].copy()
        
        # 2. 内置分词器 (针对关键词统计，需要把句子切开)
        # 注意：这里我们只用jieba做简单的切词，不需要太复杂的清洗，因为CountVectorizer有stop_words
        def tokenizer(text):
            return jieba.lcut(text)

        # 3. 准备合并后的文本
        # 将同一类的所有文本拼成一个超长字符串
        docs_per_class = docs_df.groupby(['Cluster'], as_index=False).agg({'Text': ' '.join})
        
        # 4. 使用 CountVectorizer 计算类内词频
        # 改用绝对数值避免小数据集时的参数冲突
        n_docs = len(docs_per_class)
        # 动态计算参数：min_df 最多为文档总数的 50%，max_df 留在 70% 以上
        min_df_value = max(1, n_docs // 2) if n_docs > 2 else 1
        max_df_value = max(n_docs - 1, int(n_docs * 0.7))
        
        count_vectorizer = CountVectorizer(
            tokenizer=tokenizer,
            stop_words=['的', '了', '我', '是', '在', '有', '和'], # 基础停用词
            max_df=max_df_value,  # 使用绝对值：至少出现在 70% 的簇
            min_df=min_df_value   # 使用绝对值：至少出现在 1-2 个簇
        )
        
        count_matrix = count_vectorizer.fit_transform(docs_per_class['Text'])
        words = count_vectorizer.get_feature_names_out()
        
        # 5. 提取 Top N
        keywords_dict = {}
        # count_matrix 是 (n_clusters, n_words) 的矩阵
        matrix_array = count_matrix.toarray()
        
        for i, row in docs_per_class.iterrows():
            cluster_id = row['Cluster']
            # 获取该行的词频向量
            word_counts = matrix_array[i]
            # 获取排序后的索引 (从大到小)
            sorted_indices = word_counts.argsort()[::-1]
            
            top_keywords = []
            for idx in sorted_indices[:top_n]:
                top_keywords.append(words[idx])
            
            keywords_dict[cluster_id] = ", ".join(top_keywords)
            
        return keywords_dict

    def _generate_final_report(self, texts: List[str], keyword_top_n: int, original: Optional[List] = None) -> None:
        """
        [Private] 组装最终的 DataFrame。
        结果存储于 self.results_df
        """
        # 创建临时 DataFrame
        data = {
            'Text': texts, 
            'Cluster': self._labels
        }
        
        # 如果传入了 ID，则添加到字典中
        if original is not None:
            data['Trace_ID'] = original
            
        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 提取关键词字典
        keywords_map = self._extract_cluster_keywords(df, keyword_top_n)
        
        # 映射回主表
        df['Keywords'] = df['Cluster'].map(keywords_map)
        df['Keywords'] = df['Keywords'].fillna("噪音/离群点") # 处理 -1 的情况
        
        self.results_df = df

    def save_results(self, file_path: str) -> None:
        """
        [Public] 将结果保存为 Excel 文件。
        
        Args:
            file_path (str): 输出路径，必须以 .xlsx 结尾。
        """
        if self.results_df is None:
            raise RuntimeError("请先调用 run_analysis() 方法！")
            
        self.results_df.to_excel(file_path, index=False)
        print(f"💾 结果已保存至: {file_path}")

    def get_results(self) -> pd.DataFrame:
        """
        [Public] 获取分析结果 DataFrame。
        """
        return self.results_df
    
    def get_embeddings(self) -> np.ndarray:
        """
        [Public] 获取原始 SBERT 向量。
        """
        return self._embeddings

    def save_state(self, file_path: str = "cluster_state.pkl"):
        """
        [Public] 保存当前的分析状态（核心向量和结果）。
        这样下次就不用重新跑 SBERT 编码了。
        """
        if self._embeddings is None:
            print("⚠️ 没有数据可保存，请先运行 run_analysis。")
            return

        state = {
            'embeddings': self._embeddings,           # 原始 SBERT 向量 (最宝贵的数据)
            'umap_embeddings': self._umap_embeddings, # 降维后的向量
            'labels': self._labels,                   # 聚类标签
            'results_df': self.results_df             # 最终结果表
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 分析状态已保存至: {file_path} (下次可直接加载)")

    def load_state(self, file_path: str = "cluster_state.pkl") -> bool:
        """
        [Public] 加载之前的分析状态。
        返回 True 表示加载成功，False 表示文件不存在。
        """
        if not os.path.exists(file_path):
            print(f"⚠️ 状态文件 {file_path} 不存在，将重新计算。")
            return False
            
        print(f"📂 正在加载历史状态: {file_path} ...")
        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
            
            # 恢复内部变量
            self._embeddings = state.get('embeddings')
            self._umap_embeddings = state.get('umap_embeddings')
            self._labels = state.get('labels')
            self.results_df = state.get('results_df')
            
            print("✅ 状态加载成功！已跳过 SBERT 编码步骤。")
            return True
        except Exception as e:
            print(f"❌ 加载失败，文件可能已损坏: {e}")
            return False

    def re_cluster(self, n_neighbors=15, min_cluster_size=10, keyword_top_n=5):
        """
        [Public] 基于已有的向量，重新调整聚类参数 (极速版)。
        前提：必须先 load_state 或 run_analysis。
        """
        if self._embeddings is None:
            raise ValueError("没有向量数据！请先加载状态或运行完整分析。")
            
        print("🔄 [Re-Run] 基于现有向量重新聚类...")
        # 只需要重跑后面几步，不需要重跑 _encode_text
        self._reduce_dimensions(n_neighbors, 5) # 重新降维
        self._perform_clustering(min_cluster_size) # 重新聚类
        
        # 为了生成报告，我们需要原始文本。如果 results_df 还在，可以直接取
        if self.results_df is not None:
            texts = self.results_df['Text'].tolist()
            original = None
            if 'Original_Content' in self.results_df.columns:
                original = self.results_df['Original_Content'].tolist()
            
            self._generate_final_report(texts, keyword_top_n, original)
        else:
            print("⚠️ 警告：缺少原始文本，无法生成关键词报告，仅更新了聚类标签。")

    def merge_similar_clusters(self, threshold: float = 0.9) -> None:
        """
        [Public] 后处理：自动合并语义极其相似的簇。
        基于簇中心向量的余弦相似度，将相似度高于阈值的簇进行合并。
        
        Args:
            threshold (float): 合并阈值 (0.0 - 1.0)。
                               推荐 0.90 - 0.95。
                               值越小，合并越激进（剩下的类越少）。
        """
        if self.results_df is None or self._embeddings is None:
            print("⚠️ 无法合并：缺少分析结果或向量数据。")
            return

        print(f"🔄 [Post-Process] 正在进行语义合并 (阈值: {threshold})...")
        
        # 1. 准备数据：只处理有效聚类 (忽略噪音 -1)
        valid_mask = self.results_df['Cluster'] != -1
        valid_df = self.results_df[valid_mask]
        
        if valid_df.empty:
            print("⚠️ 没有有效聚类可合并。")
            return

        # 获取所有唯一的聚类 ID
        cluster_ids = sorted(valid_df['Cluster'].unique())
        
        # 2. 计算每个簇的质心 (Centroid)
        # 质心 = 该簇所有文本向量的平均值
        centroids = []
        for cid in cluster_ids:
            # 获取该簇对应的原始向量索引
            indices = valid_df[valid_df['Cluster'] == cid].index
            vecs = self._embeddings[indices]
            centroids.append(np.mean(vecs, axis=0))
        
        if len(centroids) < 2:
            print("⚠️ 聚类数量不足 2 个，无需合并。")
            return

        # 3. 计算质心之间的相似度矩阵
        sim_matrix = cosine_similarity(centroids)
        
        # 4. 执行合并逻辑 (贪婪策略)
        # merge_map 记录映射关系: {旧ID: 新ID}
        # 初始状态每个 ID 指向自己
        merge_map = {cid: cid for cid in cluster_ids}
        merged_count = 0
        
        # 倒序遍历，优先把后面的类合并到前面的类 (保留 ID 小的)
        for i in range(len(cluster_ids) - 1, 0, -1):
            source_id = cluster_ids[i]
            
            # 寻找与当前类最相似的前序类
            best_match_idx = -1
            best_sim = -1.0
            
            for j in range(i):
                sim = sim_matrix[i][j]
                if sim > threshold and sim > best_sim:
                    best_sim = sim
                    best_match_idx = j
            
            # 如果找到了符合条件的“亲戚”
            if best_match_idx != -1:
                target_id = cluster_ids[best_match_idx]
                
                # 追溯 target_id 的最终归宿 (防止链式合并 A->B, B->C 的情况)
                while target_id != merge_map[target_id]:
                    target_id = merge_map[target_id]
                
                merge_map[source_id] = target_id
                merged_count += 1
                
                # 获取关键词用于打印日志
                src_kw = self.results_df[self.results_df['Cluster'] == source_id]['Keywords'].iloc[0].split(',')[0]
                tgt_kw = self.results_df[self.results_df['Cluster'] == target_id]['Keywords'].iloc[0].split(',')[0]
                print(f"   - 合并: C{source_id}[{src_kw}] -> C{target_id}[{tgt_kw}] (相似度: {best_sim:.3f})")

        if merged_count == 0:
            print("✅ 没有发现需要合并的簇。")
            return

        # 5. 更新 DataFrame 中的 Cluster 列
        # 使用 map 更新
        new_labels = self.results_df['Cluster'].copy()
        
        # 仅更新 merge_map 中发生变化的
        for src, tgt in merge_map.items():
            if src != tgt:
                new_labels[new_labels == src] = tgt
        
        self.results_df['Cluster'] = new_labels
        self._labels = new_labels.values # 同步更新内部 labels
        
        # 6. 因为合并了类，关键词可能会发生变化（大类涵盖了更多内容），建议重新提取关键词
        # 这里为了简单，我们暂不重新提取所有关键词，但在实际业务中建议调用一次 _extract_cluster_keywords
        # 如果你想自动刷新关键词，取消下面这行的注释：
        # self._generate_final_report(self.results_df['Text'].tolist(), keyword_top_n=5)
        
        print(f"✅ 合并完成！共减少了 {merged_count} 个微型簇。")
        print(f"   当前剩余聚类数: {len(set(new_labels)) - (1 if -1 in new_labels else 0)}")