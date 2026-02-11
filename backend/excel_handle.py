import pandas as pd
import os
from pathlib import Path

# -*- coding: utf-8 -*-


def read_excel(file_path, sheet_name=0, header=0):
    """
    读取Excel文件
    
    参数:
        file_path (str): Excel文件路径
        sheet_name (int or str): 工作表名称或索引，默认为0
        header (int): 作为列名的行号，默认为0
    
    返回:
        pd.DataFrame: 读取的数据框
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header)
        return df
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def read_multiple_sheets(file_path):
    """
    读取Excel文件的所有工作表
    
    参数:
        file_path (str): Excel文件路径
    
    返回:
        dict: 包含所有工作表的字典，键为工作表名称，值为DataFrame
    """
    try:
        sheets = pd.read_excel(file_path, sheet_name=None)
        return sheets
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None


def save_excel(df, save_path, sheet_name='Sheet1', index=False):
    """
    保存数据框为Excel文件
    
    参数:
        df (pd.DataFrame): 要保存的数据框
        save_path (str): 保存路径
        sheet_name (str): 工作表名称，默认为'Sheet1'
        index (bool): 是否保存索引，默认为False
    """
    try:
        df.to_excel(save_path, sheet_name=sheet_name, index=index)
        print(f"文件已保存到: {save_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")


def append_to_excel(file_path, df, sheet_name='Sheet1'):
    """
    追加数据到现有Excel文件
    
    参数:
        file_path (str): Excel文件路径
        df (pd.DataFrame): 要追加的数据框
        sheet_name (str): 工作表名称
    """
    try:
        existing_df = pd.read_excel(file_path, sheet_name=sheet_name)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_excel(file_path, sheet_name=sheet_name, index=False)
        print(f"数据已追加到: {file_path}")
    except Exception as e:
        print(f"追加数据失败: {e}")


def check_file_exists(file_path):
    """
    检测文件是否存在
    
    参数:
        file_path (str): 文件路径
    
    返回:
        bool: 文件存在返回True，不存在返回False
    """
    return Path(file_path).exists()

def join_cluster_summary(
    detail_file_path: str, 
    summary_file_path: str, 
    output_path: str, 
    on_key: str = 'Cluster'
):
    """
    将【所有文本详情表】与【簇概括表】合并。
    
    Args:
        detail_file_path: 包含所有数据的 Excel 路径 (必须包含 Cluster 列)
        summary_file_path: 包含簇概括的 Excel 路径 (必须包含 Cluster 列)
        output_path: 结果保存路径
        on_key: 两个表共同的关联列名，通常是 'Cluster'
    """
    print(f"🔄 正在合并文件...")
    print(f"   1. 读取详情表: {detail_file_path}")
    if not os.path.exists(detail_file_path):
        print(f"❌ 错误: 找不到详情文件")
        return

    print(f"   2. 读取概括表: {summary_file_path}")
    if not os.path.exists(summary_file_path):
        print(f"❌ 错误: 找不到概括文件")
        return
        
    try:
        df_detail = pd.read_excel(detail_file_path)
        df_summary = pd.read_excel(summary_file_path)
        
        # --- 关键检查 ---
        if on_key not in df_detail.columns:
            raise ValueError(f"详情表中找不到关联列 '{on_key}'")
        if on_key not in df_summary.columns:
            raise ValueError(f"概括表中找不到关联列 '{on_key}'")
            
        # --- 执行左连接 (Left Join) ---
        # how='left' 保证详情表的一行都不会少，只是把概括信息贴在后面
        merged_df = pd.merge(
            df_detail, 
            df_summary, 
            on=on_key, 
            how='left', 
            suffixes=('', '_Summary') # 如果有重名列，给概括表的列加后缀
        )
        
        # --- 优化列顺序 (可选) ---
        # 尝试把 'LLM_Keywords' 或 '概括' 移动到 'Cluster' 后面，方便阅读
        cols = list(merged_df.columns)
        # 假设概括表里有这些列名中的某一个
        target_cols = ['LLM_Keywords', 'LLM_Topic', 'Topic', 'Keywords']
        found_cols = [c for c in target_cols if c in cols]
        
        if found_cols:
            # 把找到的概括列挪到 Cluster 后面
            cluster_idx = cols.index(on_key)
            for c in found_cols:
                cols.remove(c)
                cols.insert(cluster_idx + 1, c)
            merged_df = merged_df[cols]

        print(f"💾 正在保存合并结果到: {output_path}")
        merged_df.to_excel(output_path, index=False)
        print(f"✅ 合并成功！总行数: {len(merged_df)}")
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")