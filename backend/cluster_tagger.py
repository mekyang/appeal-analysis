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
            
            # --- 核心逻辑：智能采样 ---
            # 1. 基础计算：取 10%
            cal_size = math.ceil(total_count * 0.1)
            
            # 2. 应用“保底”策略：取 (10% 和 5) 中的较大值
            #    这意味着：如果 10% 是 2 条，通过 max 也会强制变成 5 条
            sample_size = max(5, cal_size)
            
            # 3. 应用“物理上限”：不能超过该簇的数据总量
            #    防止出现：簇里总共只有 3 条数据，但我们试图抽 5 条的情况
            sample_size = min(sample_size, total_count)
            
            # 4. 应用“成本上限”（建议保留）：防止大簇（如 1000 条数据）抽 100 条导致 Token 爆炸
            #    即使是 10%，建议最多也不要超过 50~100 条给 LLM
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
                temperature=0.1, # 低温度，保证结果稳定
                max_tokens=50    # 限制输出长度，省钱
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ API 调用失败: {e}")
            return "提取失败"

