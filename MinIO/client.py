import torch
import torch.optim as optim
from model import Model
import boto3
from botocore.client import Config


clt = boto3.client(
    's3',
    aws_access_key_id='bPTMnjFzrOKLWtsJ99OU', 
    aws_secret_access_key='SNtqqTzrENGdcG1aXcYKe2NdDSb7GLc0dcgjeEGD',
    endpoint_url='http://localhost:9000',
)

name_of_bucket = 'hws3'
clt.upload_file(
    '/mnt/c/Users/HP/Desktop/MIPT/S7/SBT/MLOps/SBT-MLOps-2023-Fall/MinIO/model.pt',
    'hws3',
    'model.pt'  
)

clt.download_file(
    'hws3',
    'model.pt',  
    '/mnt/c/Users/HP/Desktop/MIPT/S7/SBT/MLOps/SBT-MLOps-2023-Fall/MinIO/bucket_model.pt'  
)

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

pt = torch.load('bucket_model.pt')
model.load_state_dict(pt['model_state_dict'])
opt.load_state_dict(pt['opt_state_dict'])

model.eval()

print('Model: ')
for param in model.state_dict():
    print(param, '    ', model.state_dict()[param].size())

print('Optimizer: ')
for param in opt.state_dict():
    print(param, '    ', opt.state_dict()[param])



