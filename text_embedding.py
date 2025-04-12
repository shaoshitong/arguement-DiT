import torch
import argparse
from transformers import T5Tokenizer, T5EncoderModel
import gc
import json
from PIL import Image
import os

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
        # 使用 tokenizer 批量处理所有 captions
        inputs = self.t5_tokenizer(
            captions,  # 传入所有的 captions
            padding="max_length",  # 自动填充到最大长度
            truncation=True,  # 如果超过最大长度则截断
            max_length=32,  # 设置最大长度
            return_tensors="pt",  # 返回 pytorch 的 tensor 格式
            return_attention_mask=True  # 同时返回 attention_mask
        ).to(self.t5_model.device)

        # 计算嵌入
        with torch.no_grad():
            # 传递批量的 input_ids 给模型
            outputs = self.t5_model(**inputs)
            embeddings = outputs.last_hidden_state.detach().cpu().to(torch.bfloat16)

        return embeddings  # 返回 embeddings 和 attention_mask


    def process_text_embeddings(self, captions):
        """处理文本嵌入"""

        # Compute embeddings and attention mask
        embeddings = self.compute_t5_embeddings(captions)

        # 在这里可以将 embeddings 和 attention_mask 传递给 MLP 或其他网络
        return embeddings

def load_json_data(json_path):
    """加载JSON文件并提取filename和wnid"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    # 提取filename和wnid
    extracted_data = [(item['filename'], item['wnid'], item['title'], item['tags']) for item in data]
    return extracted_data

def load_text_data(text_path):
    with open(text_path, 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    data = [(i[:len("n09421951")], i[len("n09421951")+1:]) for i in data]
    return data




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--src", help="path to source files. Must contain a 'caption_file_name.csv' file with the captions", default="/share/Rui_Huang/T2I-ImageNet-main/t2i_imagenet/data_augmentations/train/n01770393")
    # parser.add_argument("--dest", help="path to destination files", default="/path/to/output")
    # parser.add_argument("--caption_file_name", help="name of the caption file", default='captions')
    args = parser.parse_args()

    # Initialize the processor and run
    
    processor = TextEmbeddingProcessor()
    json_path = "/home/shaoshitong/project/argument-DiT/captions/imagenet-captions/caption.txt"
    extracted_data = load_text_data(json_path)
    root_path = "/data/shared_data/ILSVRC2012/caption_embeddings"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    def prompt_template(class_name):
        return f"a photo of {class_name}."
    
    for filename, class_name in extracted_data:
        prompt = prompt_template(class_name)
        embeddings = processor.process_text_embeddings(prompt)
        torch.save(embeddings, os.path.join(root_path, filename + ".pt"))
        print(f"save {filename} to {root_path}")
        
