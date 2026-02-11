import re
from abc import ABC, abstractmethod

# -------------------------------------------------------------------------
# 1. 抽象基类 (定义统一接口)
# -------------------------------------------------------------------------
class BaseContentExtractor(ABC):
    """
    内容提取抽象基类
    定义了提取器必须具备的结构，并将通用的 DataFrame 操作逻辑下沉至此。
    """

    def extract_content(self, df, column_name):
        """
        Public: 从DataFrame提取并清洗内容
        
        Args:
            df: pandas DataFrame
            column_name: 需要处理的列名
            
        Returns:
            Series: 处理后的文本列
        """
        # 确保输入列存在，避免KeyError (可选，视具体需求保留或删除)
        if column_name not in df.columns:
            return df[column_name] # 或者抛出异常
            
        text = df[column_name].astype(str) # 确保是字符串类型，防止AttributeError
        return text.apply(self._process_single_text)

    @abstractmethod
    def _process_single_text(self, text):
        """
        Abstract: 处理单条文本的具体逻辑
        必须由子类实现。
        """
        pass

class ContentExtractor_12366(BaseContentExtractor):
    """
    内容提取与清洗类 (基于关键词定位版)
    
    职责：
    1. 从混乱的文本中定位“正文”区间 (基于 '留言内容' 等关键词)。
    2. 删除无意义的垃圾短语。
    """
    
    def __init__(self):
        # 开始关键词 (保留：这是提取逻辑的核心)
        self._start_keywords = ["留言内容", "举报记录描述", "反映内容", "内容"]
        
        # 垃圾文本列表 (保留：这是去噪逻辑)
        self._trash_phrases = [
            "（税务机关联系时可附举报资料）",
            "（联系电话可以告知税务机关）",
            "（不能将联系方式告知稽查局）",
            "此工单为实名举报",
            "此工单为匿名举报",
            "举报人来电反映",
            "纳税人来电反映",
            "举报人反映",
            "纳税人反映",
            "举报人来电",
            "纳税人来电",
            "此工单无附件",
            "此工单有附件",
            "不需要回复",
            "需要回复"
        ]
    
    def _process_single_text(self, text):
        """
        Implementation: 对应原来的 _extract_and_clean
        """
        return self._extract_and_clean(text)

    def _extract_and_clean(self, text):
        """
        Private: 提取内容并执行去噪 (逻辑保持不变)
        """
        # 步骤1: 暴力提取内容区间
        content_lines = self._extract_content_brute_force(text)
        result_text = "\n".join(content_lines)
        
        # 步骤2: 后处理清洗
        result_text = self._clean_trash_phrases(result_text)
        result_text = self._clean_closing_phrases(result_text)
    
        return result_text
    
    def _extract_content_brute_force(self, text):
        """Private: 从文本中提取内容区间 (逻辑保持不变)"""
        if not isinstance(text, str):
            return []
            
        lines = text.split('\n')
        content_lines = []
        is_recording = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not is_recording:
                if any(kw in line for kw in self._start_keywords):
                    is_recording = True
                    clean_line = re.sub(
                        r"^(.*?)(留言内容|举报记录描述|反映内容|意见建议|内容)[:：]",
                        "",
                        line
                    ).strip()
                    if clean_line:
                        content_lines.append(clean_line)
                continue
            
            is_next_header = re.match(r"^.{1,100}[:：]", line)
            if is_next_header:
                break
            content_lines.append(line)
        
        return content_lines
    
    def _clean_trash_phrases(self, text):
        """Private: 清理固定垃圾短语 (逻辑保持不变)"""
        for phrase in self._trash_phrases:
            text = text.replace(phrase, "")
        return text
    
    def _clean_closing_phrases(self, text):
        """Private: 清理结尾客套话 (逻辑保持不变)"""
        pattern = r"(对此来电|希望税务).*"
        return re.sub(pattern, "", text)
    
class ContentExtractor_12345(BaseContentExtractor):
    """
    电话记录简易提取类
    
    职责：
    1. 去除开头的 "xxx来电反映" 前缀。
    2. 去除结尾的 "请处理..." 后缀。
    """

    def _process_single_text(self, text):
        """
        Implementation: 执行特定的头尾清洗逻辑
        """
        if not text:
            return ""

        # 1. 清理开头：检测句子开头有没有 "xxx来电反映"，有就删去
        # 逻辑：匹配字符串开头(^) 任意字符(.*?) 直到遇到 "来电反映"
        text = re.sub(r"^.*?来电反映", "", text)
        
        # 也可以处理可能存在的冒号或逗号残留 (可选优化)
        # text = re.sub(r"^[:：,，\s]+", "", text)

        # 2. 清理结尾：检测句子最后有无 "请处理" 开头的结尾，有就删去
        # 逻辑：匹配 "请处理" 以及其后所有字符(.*) 直到字符串结尾($)
        text = re.sub(r"请处理.*$", "", text)

        return text.strip()
    
class ContentExtractor_ZN(BaseContentExtractor):
    
    def _process_single_text(self, text):
        """
        Implementation: 直接返回原文本 (不做任何处理)
        """
        return text