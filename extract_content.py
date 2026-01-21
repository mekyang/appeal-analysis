import re

class ContentExtractor:
    """
    内容提取与清洗类 (精简版)
    
    职责：
    1. 从混乱的文本中定位“正文”区间。
    2. 删除无意义的垃圾短语（废话）。
    
    注意：脱敏功能（日期、数字、ID、公司名）已移交至 TaxDataSanitizer。
    """
    
    def __init__(self):
        # 开始关键词 (保留：这是提取逻辑的核心)
        self._start_keywords = ["留言内容", "举报记录描述", "反映内容", "内容"]
        
        # 垃圾文本列表 (保留：这是去噪逻辑，Sanitizer 不处理这些)
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
    
    def extract_content(self, df, column_name):
        """Public: 从DataFrame提取并清洗内容"""
        text = df[column_name]
        return text.apply(self._extract_and_clean)
    
    def _extract_and_clean(self, text):
        """
        Private: 提取内容并执行去噪
        """
        # 步骤1: 暴力提取内容区间 (保留)
        content_lines = self._extract_content_brute_force(text)
        result_text = "\n".join(content_lines)
        
        # 步骤2: 后处理清洗 (仅保留去噪逻辑)
        result_text = self._clean_trash_phrases(result_text)
        result_text = self._clean_closing_phrases(result_text)
    
        return result_text
    
    def _extract_content_brute_force(self, text):
        """Private: 从文本中提取内容区间"""
        # ... (这部分逻辑完全保留，代码未变，此处省略以节省篇幅) ...
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
        """Private: 清理固定垃圾短语"""
        for phrase in self._trash_phrases:
            text = text.replace(phrase, "")
        return text
    
    def _clean_closing_phrases(self, text):
        """Private: 清理结尾客套话"""
        pattern = r"(对此来电|希望税务).*"
        return re.sub(pattern, "", text)
