import torch
import onnx
import onnxruntime as ort
from transformers import BertModel, BertTokenizer
#import tensorrt as trt
import numpy as np

# https://huggingface.co/transformers/v3.0.2/model_doc/bert.html
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_text = "Hi, my name is"
input_ids = torch.tensor([tokenizer.encode(input_text)])

onnx_model_path = "bert_model.onnx"
dummy_input = torch.tensor(input_ids)

torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=11)