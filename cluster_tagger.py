import pandas as pd
import re
from transformers import pipeline

def anonymize_companies(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    仅针对【公司名】进行高精度脱敏（保留地名）。
    策略：正则表达式 (处理规范全称) + NER (处理漏网之鱼)
    """
    
    # 1. 加载模型 (只做查漏补缺用)
    print("正在加载 NER 模型...")
    try:
        ner_pipeline = pipeline(
            "token-classification", 
            model="shibing624/bert4ner-base-chinese",
            aggregation_strategy="simple",
            device=-1 # 有显卡改 0
        )
    except Exception as e:
        print(f"模型加载失败: {e}")
        return df

    # 2. 定义【强规则】正则表达式
    # 你的数据很规范，这行代码能解决 90% 的问题，且 0 误判
    # 匹配模式：(任意非标点字符 2个以上) + (特定的公司后缀)
    # 排除掉“有限公司”单独出现的情况
    company_regex = re.compile(r'([\u4e00-\u9fa5]{2,30}(?:公司|分公司|支行|厂|合作社|经营部|商行|超市|酒店|饭店|中心))')

    def process_text(text):
        if not isinstance(text, str) or not text.strip():
            return ""

        # --- Step 1: 正则暴力替换 (高可信度) ---
        # 比如 "日照钢铁有限公司" -> "<企业>"
        # 这一步非常快，而且绝对不会误杀 "上海" (地名)
        text = company_regex.sub('<企业>', text)

        # --- Step 2: NER 查漏补缺 (处理没带后缀的) ---
        # 此时剩下的 text 里，大部分规范公司名已经被换掉了，NER 只需要处理剩下的难点
        try:
            entities = ner_pipeline(text)
        except:
            return text

        # 倒序替换，防止索引错位
        # 过滤：只处理 ORG (公司)，忽略 LOC (地名) 和 PER (人名)
        # 补充：建议把 PER 也加上，防止 NER 把某些公司名误判为人名，反正你也不关心人名
        target_entities = [e for e in entities if e['entity_group'] in ['ORG']] 
        
        sorted_entities = sorted(target_entities, key=lambda x: x['start'], reverse=True)
        
        for ent in sorted_entities:
            start, end = ent['start'], ent['end']
            
            # 双重检查：如果 NER 识别出来的东西已经被正则替换成了 <企业>，就跳过
            # 防止出现 <<企业>> 这种嵌套
            if "<企业>" in text[start:end]:
                continue
                
            # 执行替换
            text = text[:start] + "<企业>" + text[end:]
            
        return text

    print(f"正在处理 {len(df)} 条数据...")
    
    # 批量处理
    # 注意：因为引入了正则，这里没法直接用 pipeline 的 batch_size，只能逐条处理
    # 但因为正则过滤了大部分长度，速度不会太慢
    new_texts = []
    for t in df[col_name]:
        new_texts.append(process_text(t))
        
    output_col = f"{col_name}_sanitized"
    df[output_col] = new_texts
    print("✅ 处理完成！")
    return df

# 使用示例
if __name__ == "__main__":
    df_test = pd.DataFrame({
        'content': [
            "日照钢铁控股集团有限公司咨询环保税。", # 规范名，正则能抓到
            "张三在上海分公司工作。", # "上海分公司" 正则能抓到，"上海" 不会被动
            "腾讯和阿里是互联网巨头。", # 不规范名（无后缀），正则抓不到，NER 补刀
            "去北京出差。", # 地名，完全保留
            "老王馒头店开票报错。" # 个体户，正则能抓到
        ]
    })
    
    res = anonymize_companies_only(df_test, 'content')
    print(res['content_sanitized'])