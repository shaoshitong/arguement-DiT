import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np

class TextEmbeddingProcessor:
    def __init__(self, model_name='google/flan-t5-xl'):
        self.model_name = model_name
        self.t5_model, self.t5_tokenizer = self.load_t5_model()  # 在初始化时加载模型

    def load_t5_model(self):
        """加载T5模型及其tokenizer"""
        tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        model = T5EncoderModel.from_pretrained(self.model_name)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return model, tokenizer

    def compute_t5_embeddings(self, captions):
        """计算T5模型的嵌入"""
        # 使用 tokenizer 批量处理所有 captions
        inputs = self.t5_tokenizer(
            captions,  # 传入所有的 captions
            padding="max_length",  # 自动填充到最大长度
            truncation=True,  # 如果超过最大长度则截断
            max_length=7,  # 设置最大长度
            return_tensors="pt",  # 返回 pytorch 的 tensor 格式
            return_attention_mask=True  # 同时返回 attention_mask
        ).to(self.t5_model.device)

        # 计算嵌入
        with torch.no_grad():
            # 传递批量的 input_ids 给模型
            outputs = self.t5_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # 对每个句子取平均作为嵌入

        return embeddings  # 返回 embeddings 和 attention_mask


    def process_text_embeddings(self, captions):
        """处理文本嵌入"""
        embeddings = self.compute_t5_embeddings(captions)
        return embeddings




def process_and_save_embeddings(root_dir, processor):
    # 遍历目录中的每个类别文件夹
    for class_folder in os.listdir(root_dir):
        class_folder_path = os.path.join(root_dir, class_folder)

        # 如果是目录且包含 captions.csv 文件
        if os.path.isdir(class_folder_path):
            caption_file = os.path.join(class_folder_path, 'single_caption.npy')
            if os.path.isfile(caption_file):
                print(f"Processing captions for {class_folder}")
                
                # 读取captions
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption = np.load(caption_file, allow_pickle=True).item()

                # 计算嵌入
                embeddings = processor.process_text_embeddings(caption)

                # 保存嵌入到 CSV 文件
                embeddings_file = os.path.join(class_folder_path, 'caption_embedding.npy')
                np.save(embeddings_file, embeddings)
                print(f"✅ Saved embedding to {embeddings_file}")


if __name__ == "__main__":
    root_dir = "/data/shared_data/ILSVRC2012/train"  # 数据所在目录
    processor = TextEmbeddingProcessor()  # 初始化处理器
    process_and_save_embeddings(root_dir, processor)  # 处理并保存嵌入
