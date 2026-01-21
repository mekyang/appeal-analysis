import pandas as pd
import math
from openai import OpenAI  # 适配 DeepSeek, Kimi, GPT 等所有兼容 OpenAI 协议的模型

class LLMKeywordExtractor:
    """
    LLM 关键词提取器
    
    功能：
    1. 按簇随机抽取 10% 数据。
    2. 利用大模型 API 归纳提取核心业务关键词。
    """

    def __init__(self, api_key: str, base_url: str, model: str = "deepseek-chat"):
        """
        Args:
            api_key: 模型 API Key
            base_url: 模型 API 地址 (如 https://api.deepseek.com)
            model: 模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def extract_keywords(self, df: pd.DataFrame, text_col: str, cluster_col: str = 'Cluster') -> pd.DataFrame:
        """
        执行提取流程
        
        Args:
            df: 包含聚类结果的 DataFrame
            text_col: 文本列名 (建议使用脱敏后的列)
            cluster_col: 聚类 ID 列名
            
        Returns:
            pd.DataFrame: 新增了 'LLM_Keywords' 列的 DataFrame
        """
        print(f"🤖 [LLM] 开始对 {len(df[cluster_col].unique())} 个聚类进行关键词提取...")
        
        # 1. 准备结果字典 {cluster_id: keywords}
        cluster_keywords_map = {}
        
        # 2. 获取所有有效簇 (排除噪音 -1，或者你也想处理噪音?)
        # 通常建议排除 -1，因为噪音里没有共性
        unique_clusters = sorted(df[cluster_col].unique())
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            cluster_keywords_map[-1] = "噪音数据/杂项"

        # 3. 遍历每个簇
        for cid in unique_clusters:
            # 获取该簇所有数据
            cluster_data = df[df[cluster_col] == cid]
            total_count = len(cluster_data)
            
            # --- 核心逻辑：随机抽取 10% ---
            # math.ceil 向上取整，保证至少有 1 条
            sample_size = math.ceil(total_count * 0.1)
            
            # 设定上限（可选）：如果一个类有 1万条，10%就是1000条，这会把 Token 撑爆且费钱
            # 建议加个硬顶，比如最多取 50 条，这足够代表性了
            sample_size = min(sample_size, 50) 
            
            sampled_texts = cluster_data[text_col].sample(n=sample_size, random_state=42).tolist()
            
            print(f"   - Processing Cluster {cid} (Total: {total_count}, Sampled: {sample_size})...")
            
            # 4. 调用 LLM
            keywords = self._call_llm_api(sampled_texts)
            cluster_keywords_map[cid] = keywords
            
        # 5. 回填结果
        # 使用 map 函数将字典值映射回 DataFrame
        df['LLM_Keywords'] = df[cluster_col].map(cluster_keywords_map)
        
        print("✅ 关键词提取完成！")
        return df

    def _call_llm_api(self, texts: list) -> str:
        """
        [Private] 构造 Prompt 并调用 API
        """
        # 拼接文本，用换行符分隔
        context_str = "\n".join([f"- {t}" for t in texts])
        
        # --- Prompt 设计 ---
        # 技巧：指定输出格式，禁止废话
        prompt = f"""
你是一名资深的税务数据分析师。以下是从某类税务咨询工单中随机抽取的若干条记录。
请分析这些文本的共同主题，提取 3-5 个核心业务关键词。

要求：
1. 关键词必须精准概括核心业务实体或动作（如：个税APP, 申报失败, 密码重置）。
2. 不要包含“咨询”、“纳税人”、“问题”等无意义的通用词。
3. 直接输出关键词，用逗号分隔。不要输出任何解释性文字。

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
                temperature=0.1, # 低温度，保证结果稳定
                max_tokens=50    # 限制输出长度，省钱
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return "提取失败"

