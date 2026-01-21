import pandas as pd
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