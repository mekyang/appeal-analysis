import re
import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # 进度条库

class TaxDataSanitizer:
    def __init__(self, use_ner=True, device=0, batch_size=64):
        """
        Args:
            use_ner (bool): 是否启用 BERT。开启后将不再使用“公司名正则”，完全依赖模型。
            device (int): 显卡 ID。
            batch_size (int): 批处理大小，默认 64。
        """
        self.use_ner = use_ner
        self.device = device
        self.batch_size = batch_size
        
        # --- 1. 通用正则 (无论开不开 NER 都要跑) ---
        self.regex_date = re.compile(
            r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|'
            r'\d{4}-\d{1,2}-\d{1,2}|\d{4}\.\d{1,2}\.\d{1,2}|20\d{2}年)'
        )
        self.regex_id = re.compile(r'(?<![<a-zA-Z])([a-zA-Z0-9]{10,})(?![>a-zA-Z])')
        self.regex_number = re.compile(r'(?<![<a-zA-Z0-9])(\d+\.?\d*)(?![>a-zA-Z0-9])')

        # --- 2. 备用正则 (仅当 use_ner=False 时才使用) ---
        self.regex_company_fallback = re.compile(
            r'([\u4e00-\u9fa5]{2,30}(?:公司|分公司|支行|厂|合作社|经营部|商行|超市|酒店|饭店|事务?所))'
        )

        # --- 3. 加载模型 ---
        if self.use_ner:
            print(f"🚀 Loading NER model on device cuda:{device}...")
            try:
                # 优先从本地目录加载模型和分词器，避免被当作 HuggingFace repo id 解析
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                import torch

                local_model_dir = "models/models--shibing624--bert4ner-base-chinese"
                tokenizer = AutoTokenizer.from_pretrained(local_model_dir, use_fast=True)
                model = AutoModelForTokenClassification.from_pretrained(local_model_dir)

                if torch.cuda.is_available():
                    try:
                        model.to(torch.device(f"cuda:{device}"))
                    except Exception:
                        pass

                self.ner_pipeline = pipeline(
                    "token-classification",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=device
                )
            except Exception as e:
                print(f"⚠️ Model load failed, fallback to REGEX ONLY mode: {e}")
                self.use_ner = False

    def _common_preprocess(self, text):
        """通用预处理：仅处理日期和ID，绝对不碰公司名"""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self.regex_date.sub('<日期>', text)
        text = self.regex_id.sub('<识别号>', text)
        return text

    def _apply_ner_logic(self, text, entities):
        """应用 NER 实体替换逻辑"""
        if not entities:
            return text
            
        # 筛选 ORG 和 PER
        valid_ents = [e for e in entities if e['entity_group'] in ['ORG', 'PER']]
        # 倒序替换防止索引偏移
        valid_ents.sort(key=lambda x: x['start'], reverse=True)
        
        for ent in valid_ents:
            start, end = ent['start'], ent['end']
            span = text[start:end]
            
            # 保护机制：如果 NER 抓到了刚才正则生成的 <日期> 等标签，跳过
            if '<' in span and '>' in span:
                continue
                
            replacement = '<企业名>' if ent['entity_group'] == 'ORG' else '<人名>'
            text = text[:start] + replacement + text[end:]
        return text

    def process_dataframe(self, df, col_name, output_col=None, progress_callback=None):
        """
        保持接口不变，内部根据配置自动选择最优路径
        
        Args:
            progress_callback (callable): 可选的进度回调函数，签名为 progress_callback(current, total, stage)
                                        用于在 Streamlit 或其他前端显示进度
        """
        if output_col is None:
            output_col = f"{col_name}_sanitized"

        print(f"⚡ Processing {len(df)} rows. Logic: {'[NER Model Only]' if self.use_ner else '[Regex Only]'} for Companies.")
        
        if progress_callback:
            progress_callback(0, len(df), "初始化预处理...")

        # 1. 提取数据并进行通用预处理 (Date + ID)
        texts = df[col_name].apply(self._common_preprocess).tolist()

        # 2. 分支逻辑
        if self.use_ner:
            # === 分支 A: 使用 GPU NER (不跑正则公司匹配) ===
            final_texts = []
            print(f"Running GPU Batch Inference (Batch Size={self.batch_size})...")
            
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            
            # 批量循环
            for batch_idx, i in enumerate(range(0, len(texts), self.batch_size)):
                batch = texts[i : i + self.batch_size]
                try:
                    # 显式传递 batch_size 优化推理
                    batch_results = self.ner_pipeline(batch, batch_size=self.batch_size)
                except Exception:
                    batch_results = [[] for _ in batch]

                # 替换实体
                for text, entities in zip(batch, batch_results):
                    final_texts.append(self._apply_ner_logic(text, entities))
                
                # 更新进度回调
                if progress_callback:
                    current_count = min(i + self.batch_size, len(texts))
                    progress_callback(current_count, len(texts), f"NER推理中 ({batch_idx + 1}/{total_batches})...")
            
            texts = final_texts

        else:
            # === 分支 B: 仅使用正则 (回退逻辑) ===
            print("Running Regex Fallback for Companies...")
            # 使用列表推导式加速
            texts = [self.regex_company_fallback.sub('<企业>', t) for t in texts]
            
            if progress_callback:
                progress_callback(len(texts), len(texts), "正则表达式处理完成...")

        # 3. 通用收尾 (数字)
        print("Finalizing numbers...")
        texts = [self.regex_number.sub('<数字>', t) for t in texts]
        
        if progress_callback:
            progress_callback(len(texts), len(texts), "数字标记完成...")

        # 4. 回填
        df[output_col] = texts
        print("✅ Done.")
        return df

    def sanitize_text(self, text):
        """单条处理接口 (逻辑保持一致)"""
        text = self._common_preprocess(text)
        
        if self.use_ner:
            # NER 路径：绝不跑 regex_company
            try:
                entities = self.ner_pipeline(text)
                text = self._apply_ner_logic(text, entities)
            except:
                pass
        else:
            # 正则路径
            text = self.regex_company_fallback.sub('<企业>', text)
            
        text = self.regex_number.sub('<数字>', text)
        return text