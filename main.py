from config import *
from extract_content import ContentExtractor
from data_analysis import TaxClusteringEngine
from cluster_eval import ClusterEvaluator
from excel_handle import *
import pandas as pd

content_extractor = ContentExtractor()

def deal_data():
    df = read_excel(EXCEL_PATH, sheet_name=0, header=0)

    # 提取内容
    extracted_contents = content_extractor.extract_content(df, '业务内容')

    # 保存结果到新的Excel文件
    output_df = pd.DataFrame({'Extracted Content': extracted_contents})
    save_excel(output_df, SAVE_PATH)


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
        #results_df = clustering_engine.run_analysis(text)
        clustering_engine.save_state(STATE_FILE)

    # 保存分析结果
    # cluster_eval = ClusterEvaluator(df=results_df, embeddings=clustering_engine.get_embeddings())
    # cluster_eval.run_full_report()
    # clustering_engine.save_results(OUTPUT_DIR)

    clustering_engine.merge_similar_clusters(threshold=0.92)
    clustering_engine.save_results('最终结果_已合并.xlsx')

if __name__ == "__main__":
    main()

