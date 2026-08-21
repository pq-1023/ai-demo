from pypdf import PdfReader
import os
import re


def clean_text(raw_text: str) -> str:
    """
    文档数据清洗
    1.移除制表符
    2.合并多余空行（保留段落结构）
    3.去除行首尾空白，合并行内多余空格
    """
    text = raw_text.replace("\t", " ")
    # 多个空格合并（同一行内）
    text = re.sub(r"[^\S\n]+", " ", text)
    # 按行处理：首尾去空白
    lines = [line.strip() for line in text.split("\n")]
    # 合并多余空行：连续3+个空行→1个空行，保留段落之间的单个换行
    result_lines = []
    empty_count = 0
    for line in lines:
        if line == "":
            empty_count += 1
            if empty_count <= 1:
                result_lines.append("")
        else:
            empty_count = 0
            result_lines.append(line)
    text = "\n".join(result_lines)
    text = text.strip()
    return text

#split_text () 分片函数，以段落为单位切分保证语义完整性
def split_text(long_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # 第一阶段：按段落切分（空行作为段落分隔）
    paragraphs = re.split(r'\n\s*\n', long_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # 第二阶段：将段落按 chunk_size 拼接，超长段落内部再按句号切分
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        # 如果段落本身超长，在段落内按句号进一步切分
        if len(para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            # 按句号切分长段落
            sens = re.split(r'(?<=[。！？.!？！])\s*', para)
            for s in sens:
                s = s.strip()
                if not s:
                    continue
                if len(current_chunk) + len(s) <= chunk_size:
                    current_chunk += s
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    if len(s) > chunk_size:
                        chunks.append(s)
                        current_chunk = ""
                    else:
                        current_chunk = s
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = ""
        else:
            if len(current_chunk) + len(para) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
    if current_chunk:
        chunks.append(current_chunk)

    # 第三阶段：基于 overlap 做相邻分片重叠（从句子边界开始，避免语义断裂）
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            prev = chunks[i - 1]
            if len(prev) > chunk_overlap:
                tail = prev[-chunk_overlap:]
                boundaries = [m.end() for m in re.finditer(r'[。！？.!？！\n]', tail)]
                if boundaries:
                    overlap_text = tail[boundaries[-1]:]
                else:
                    overlap_text = tail
            else:
                overlap_text = prev
            overlapped.append(overlap_text + c)
    return overlapped


def load_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text
        # PDF读取完成后执行清洗
        clean_result = clean_text(full_text)
        if len(clean_result) == 0:
            raise RuntimeError("PDF提取后无有效文本内容")
        return clean_result
    except Exception as e:
        raise RuntimeError(f"PDF读取失败:{e}")


def load_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        clean_result = clean_text(raw)
        if len(clean_result) == 0:
            raise RuntimeError("TXT文档无有效文本")
        return clean_result
    except UnicodeDecodeError:
        raise RuntimeError("txt编码错误，请使用utf‑8编码")
    except Exception as e:
        raise RuntimeError(f"TXT读取失败:{e}")


def load_document(file_path: str) -> str:
    suffix = os.path.splitext(file_path)[1].lower()
    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".txt":
        return load_txt(file_path)
    else:
        raise Exception("仅支持PDF、TXT文档")