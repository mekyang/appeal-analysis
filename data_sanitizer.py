import re
import pandas as pd
from transformers import pipeline

class TaxDataSanitizer:
    """
    税务数据脱敏专用类
    功能：对公司名、日期、长识别号（身份证/税号/工单）、普通数字进行清洗
    """

    def __init__(self, use_ner=True, device=-1):
        """
        Args:
            use_ner (bool): 是否启用 BERT 模型进行公司名查漏补缺（建议开启，但会消耗内存）。
            device (int): 运行设备，-1 为 CPU，0 为 GPU。
        """
        self.use_ner = use_ner
        
        # 1. 预编译正则模式 (提升速度)
        
        # [日期]：覆盖常见年月日格式
        self.regex_date = re.compile(
            r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|'
            r'\d{4}-\d{1,2}-\d{1,2}|\d{4}\.\d{1,2}\.\d{1,2}|20\d{2}年)'
        )

        # [识别号]：匹配 工单号、税号、身份证号
        # 逻辑：匹配10位以上的连续数字或字母组合 (身份证18位，税号15-20位，工单通常长数字)
        # 排除掉已经被替换的 <日期> 等标签防止误伤
        self.regex_id = re.compile(r'(?<![<a-zA-Z])([a-zA-Z0-9]{10,})(?![>a-zA-Z])')

        # [公司名-强规则]：匹配规范后缀
        self.regex_company = re.compile(
            r'([\u4e00-\u9fa5]{2,30}(?:公司|分公司|支行|厂|合作社|经营部|商行|超市|酒店|饭店|事务?所))'
        )

        # [数字]：匹配剩余的短数字 (金额、数量等)
        # 排除已有的标签 <...>
        self.regex_number = re.compile(r'(?<![<a-zA-Z0-9])(\d+\.?\d*)(?![>a-zA-Z0-9])')

        # 2. 加载 NER 模型 (仅当需要时)
        if self.use_ner:
            print("正在加载 NER 模型 (bert4ner-base-chinese)...")
            try:
                self.ner_pipeline = pipeline(
                    "token-classification",
                    model="shibing624/bert4ner-base-chinese",
                    aggregation_strategy="simple",
                    device=device
                )
            except Exception as e:
                print(f"⚠️ 模型加载失败，将降级为仅使用正则模式: {e}")
                self.use_ner = False

    def sanitize_text(self, text):
        """处理单条文本"""
        if not isinstance(text, str) or not text.strip():
            return ""

        # --- Step 1: 替换日期 (优先级最高，防止里面的数字干扰) ---
        text = self.regex_date.sub('<日期>', text)

        # --- Step 2: 替换长识别号 (身份证/税号/工单) ---
        # 这些通常由长数字/字母组成，必须在处理普通数字前处理
        text = self.regex_id.sub('<识别号>', text)

        # --- Step 3: 正则替换公司名 (高可信度) ---
        text = self.regex_company.sub('<企业>', text)

        # --- Step 4: NER 模型查漏补缺 (处理不规范公司名) ---
        if self.use_ner:
            text = self._apply_ner_patch(text)

        # --- Step 5: 替换剩余普通数字 ---
        text = self.regex_number.sub('<数字>', text)

        return text

    def _apply_ner_patch(self, text):
        """内部方法：使用 NER 补充识别正则没抓到的公司名"""
        try:
            # 只有当文本中没有被正则完全覆盖时才跑 NER，节省资源
            entities = self.ner_pipeline(text)
        except:
            return text

        # 筛选出 ORG (机构)，忽略 LOC (地名) 和 PER (人名)
        # 如果你希望把人名也替换，可以在列表里加上 'PER'，并修改 replacement
        target_entities = [e for e in entities if e['entity_group'] == 'ORG']
        
        # 倒序替换
        sorted_entities = sorted(target_entities, key=lambda x: x['start'], reverse=True)

        for ent in sorted_entities:
            start, end = ent['start'], ent['end']
            span = text[start:end]

            # 关键保护：
            # 1. 如果 NER 抓到的内容已经被正则变成了 <企业>，跳过
            # 2. 如果 NER 抓到的包含 <识别号> 或 <日期> (模型有时候会发疯)，跳过
            if '<' in span and '>' in span:
                continue
            
            # 执行替换
            text = text[:start] + '<企业名>' + text[end:]
        
        return text

    def process_dataframe(self, df, col_name, output_col=None):
        """批量处理 DataFrame"""
        print(f"正在处理 {len(df)} 条数据...")
        
        # 如果不开 NER，可以直接用 apply，速度很快
        # 如果开 NER，为了进度条或稳定性，建议循环或分批
        
        # 这里演示简单方式
        if output_col is None:
            output_col = f"{col_name}_sanitized"
            
        df[output_col] = df[col_name].apply(self.sanitize_text)
        
        print("✅ 脱敏处理完成！")
        return df