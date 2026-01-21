from config import *
from extract_content import ContentExtractor
from data_analysis import TaxClusteringEngine
from cluster_eval import ClusterEvaluator
from data_sanitizer import TaxDataSanitizer
from cluster_tagger import LLMKeywordExtractor
from excel_handle import *
import pandas as pd

content_extractor = ContentExtractor()

# def deal_data():
#     df = read_excel(EXCEL_PATH, sheet_name=0, header=0)

#     # 提取内容
#     extracted_contents = content_extractor.extract_content(df, '业务内容')

#     # 保存结果到新的Excel文件
#     output_df = pd.DataFrame({'Extracted Content': extracted_contents})
#     save_excel(output_df, SAVE_PATH)

def deal_data():
    print(f"📂 开始读取原始文件: {EXCEL_PATH} ...")
    try:
        # 1. 读取原始数据
        df = read_excel(EXCEL_PATH, sheet_name=0, header=0)
        df['Trace_ID'] = df.index + 2 
        print(f"✅ 已生成追踪序号 (Trace_ID)，范围: {df['Trace_ID'].min()} - {df['Trace_ID'].max()}")
        
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # --- 初始化工具链 ---
    print("🔧 初始化处理工具...")
    extractor = ContentExtractor() 
    sanitizer = TaxDataSanitizer(use_ner=True) 

    print("1️⃣ 正在提取正文并去除废话...")
    raw_content_series = extractor.extract_content(df, '业务内容')

    print("2️⃣ 正在进行隐私脱敏...")
    temp_df = pd.DataFrame({'temp_text': raw_content_series})
    sanitizer.process_dataframe(temp_df, 'temp_text')

    output_df = pd.DataFrame({
        'Trace_ID': df['Trace_ID'],              
        'Original_Content': df['业务内容'],      
        'Extracted_Content': temp_df['temp_text_sanitized'] 
    })
    
    initial_count = len(output_df)
    output_df = output_df[output_df['Extracted_Content'].str.strip() != '']
    final_count = len(output_df)
    
    print(f"🧹 已过滤无效空行: {initial_count} -> {final_count} (删除了 {initial_count - final_count} 条)")

    print(f"💾 正在保存结果到: {SAVE_PATH} ...")
    save_excel(output_df, SAVE_PATH)
    print("✅ 数据预处理完毕！")

def main():
    # 读取Excel文件
    if not check_file_exists(SAVE_PATH):
        print("提取内容文件不存在，正在生成...")
        deal_data()
    
    try:
        df = read_excel(SAVE_PATH, sheet_name=0, header=0)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return

    # 创建 TaxClusteringEngine 实例
    clustering_engine = TaxClusteringEngine(EMBEDDING_MODEL_NAME)

    # 执行分析
    text = df[DATA_COLUMN].dropna().astype(str).tolist()
    text = [t for t in text if len(t.strip()) > 0]

    print(f"有效文本数量: {len(text)}")

    load_cluster = input("是否加载之前的聚类状态？(y/n): ").strip().lower()
    if load_cluster == 'y':
        # 尝试加载之前的状态
        if clustering_engine.load_state(STATE_FILE):
            results_df = clustering_engine.get_results()
        else:
            results_df = clustering_engine.run_analysis(text)
            clustering_engine.save_state(STATE_FILE)

    else:
        # 如果没有状态文件，则运行完整分析流程          
        results_df = clustering_engine.run_analysis(text)
        clustering_engine.save_state(STATE_FILE)

    #保存分析结果
    cluster_eval = ClusterEvaluator(df=results_df, embeddings=clustering_engine.get_embeddings())
    cluster_eval.run_full_report()
    clustering_engine.save_results(OUTPUT_DIR)

    # clustering_engine.merge_similar_clusters(threshold=0.92)
    # clustering_engine.save_results('最终结果_已合并.xlsx')

if __name__ == "__main__":
    # 模拟数据
    data = read_excel(OUTPUT_DIR)
    df = pd.DataFrame(data)

    # 1. 初始化 (替换为你的 Key)
    # 推荐使用 DeepSeek，便宜且中文能力强
    extractor = LLMKeywordExtractor(
        api_key="sk-f1fec6b90628475ba7ce12b2c389c85a", 
        base_url="https://api.deepseek.com"
    )

    # 2. 运行提取
    df_result = extractor.extract_keywords(df, text_col='Text')

    # 3. 查看结果
    print(df_result[['Cluster', 'LLM_Keywords']].drop_duplicates())

