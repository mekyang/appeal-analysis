import re

class ContentExtractor:
    """内容提取与清洗类"""
    
    def __init__(self):
        # 开始关键词
        self._start_keywords = ["留言内容", "举报记录描述", "反映内容", "内容"]
        
        # 垃圾文本列表
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
        
        # 日期正则模式
        self._date_patterns = [
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"\d{4}年\d{1,2}月",
            r"\d{1,2}月\d{1,2}日",
            r"\d{4}-\d{1,2}-\d{1,2}",
            r"\d{4}\.\d{1,2}\.\d{1,2}",
            r"\d{1,2}:\d{1,2}",
            r"20\d{2}年",
        ]
    
    def extract_content(self, df, column_name):
        """
        Public: 从DataFrame提取并清洗内容
        
        Args:
            df: DataFrame对象
            column_name: 列名
            
        Returns:
            Series: 清洗后的文本Series
        """
        text = df[column_name]
        return text.apply(self._extract_and_clean)
    
    def _extract_and_clean(self, text):
        """
        Private: 提取内容并执行完整清洗流程
        
        Args:
            text: 输入文本
            
        Returns:
            str: 清洗后的文本
        """
        # 步骤1: 暴力提取内容区间
        content_lines = self._extract_content_brute_force(text)
        result_text = "\n".join(content_lines)
        
        # 步骤2: 后处理清洗
        result_text = self._clean_work_order(result_text)
        result_text = self._clean_dates(result_text)
        result_text = self._clean_trash_phrases(result_text)
        result_text = self._clean_closing_phrases(result_text)
        result_text = self._normalize_numbers(result_text)
        
        return result_text
    
    def _extract_content_brute_force(self, text):
        """
        Private: 从文本中提取内容区间（从关键词开始到下一个表头结束）
        
        Args:
            text: 输入文本
            
        Returns:
            list: 内容行列表
        """
        lines = text.split('\n')
        content_lines = []
        is_recording = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 判断是否开始录制
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
            
            # 判断是否该停止（安全锁逻辑）
            is_next_header = re.match(r"^.{1,100}[:：]", line)
            if is_next_header:
                break
            
            content_lines.append(line)
        
        return content_lines
    
    def _clean_work_order(self, text):
        """Private: 清理工单编号"""
        work_order_pattern = r"(原)?工单编号(为)?[：:]?"
        return re.sub(work_order_pattern, "", text)
    
    def _clean_dates(self, text):
        """Private: 清理日期相关文本"""
        for pattern in self._date_patterns:
            text = re.sub(pattern, "", text)
        return text
    
    def _clean_trash_phrases(self, text):
        """Private: 清理固定垃圾短语"""
        for phrase in self._trash_phrases:
            text = text.replace(phrase, "")
        return text
    
    def _clean_closing_phrases(self, text):
        """Private: 清理结尾客套话"""
        pattern = r"(对此来电|希望税务).*"
        return re.sub(pattern, "", text)
    
    def _normalize_numbers(self, text):
        """
        Private: 数字归一化处理
        - 长数字串(10位以上)替换为<ID>
        - 普通数字替换为<NUM>，但保留百分比
        """
        text = re.sub(r"\d{10,}", "<ID>", text)
        text = re.sub(r"\d+(?!%)", "<NUM>", text)
        return text