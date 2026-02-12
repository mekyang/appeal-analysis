import pandas as pd
import math
from openai import OpenAI

class LLMKeywordExtractor:
    """
    LLM 关键词提取器
    """

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract_keywords(self, df: pd.DataFrame, text_col: str, cluster_col: str = 'Cluster', progress_callback=None) -> pd.DataFrame:
        """
        Args:
            progress_callback: 回调函数 func(current, total, message)
        """
        print(f"🤖 [LLM] 开始对 {len(df[cluster_col].unique())} 个聚类进行关键词提取...")
        
        cluster_keywords_map = {}
        unique_clusters = sorted(df[cluster_col].unique())
        
        # 排除噪音
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            cluster_keywords_map[-1] = "噪音数据/杂项"

        total_clusters = len(unique_clusters)

        # 遍历每个簇
        for idx, cid in enumerate(unique_clusters):
            # --- 汇报进度 ---
            if progress_callback:
                # 进度条计算：当前是第几个 / 总数
                progress_callback(idx, total_clusters, f"正在分析第 {cid} 类话题...")

            cluster_data = df[df[cluster_col] == cid]
            total_count = len(cluster_data)
            
            # 智能采样逻辑
            cal_size = math.ceil(total_count * 0.1)
            sample_size = max(5, cal_size)
            sample_size = min(sample_size, total_count)
            sample_size = min(sample_size, 50) 
            
            sampled_texts = cluster_data[text_col].sample(n=sample_size, random_state=42).tolist()
            
            # 调用 LLM
            keywords = self._call_llm_api(sampled_texts)
            cluster_keywords_map[cid] = keywords
            
        # 进度 100%
        if progress_callback:
            progress_callback(total_clusters, total_clusters, "所有话题分析完成，正在整理...")

        # 回填结果
        df['LLM_Keywords'] = df[cluster_col].map(cluster_keywords_map)
        return df

    def _call_llm_api(self, texts: list) -> str:
        # ... (保持原有的 Prompt 和调用逻辑不变，这里为了节省篇幅省略 Prompt 内容，请保留你原文件中的 Prompt) ...
        context_str = "\n".join([f"- {t}" for t in texts])
        prompt = f"""
你是一名资深的税务数据分析师。某类税务咨询工单进行了聚类，以下是其中一个簇的几个内容文本。
请分析这些文本的共同主题以概括此聚类簇所代表的含义。

要求：
1. 必须精准概括核心业务实体或动作（举报企业偷税漏税、虚开发票）。
2. 文本必须充分概括出特点，体现出差异性，比如不要直接说拒开发票，而应该具体的说该簇在什么情况下或什么实体在拒开发票。
3. 概括文本不要太短，要形成一个句子的形式。
4. 不要包含“咨询”、“纳税人”、“问题”等无意义的通用词。
5. 直接输出概括文本。不要输出任何解释性文字。

样本数据：
{context_str}

关键词：
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个只输出结果的关键词提取机器。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return "提取失败"