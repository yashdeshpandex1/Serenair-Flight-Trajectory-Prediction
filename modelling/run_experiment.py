import torch 
import torch.nn as nn
from .model_1 import LSTMModelV1
from .model_2 import LSTMModelV2
from .model_3 import GRUModel
from .model_4 import BidirectionalLSTMModelV1
from .model_5 import HybridConvLSTMModelV1
from .model_6 import HybridConvLSTMModelV2
from .model_7 import HybridConvLSTMV3
from .model_8 import HybridConvLSTMBidirectionalModel
from .model_9 import AttentionModel
from .model_10 import HybridConvAttentionGRUModel
from .model_11 import Seq2SeqTrajectoryLSTMV1
from .model_12 import Seq2SeqTrajectoryLSTMV2

MODEL_REGISTRY = {
    'LSTMModelV1': LSTMModelV1,
    'LSTMModelV2': LSTMModelV2,
    'GRUV1': GRUModel,
    'BidirectionalLSTMV1': BidirectionalLSTMModelV1,
    'HybridConvLSTMV1': HybridConvLSTMModelV1,
    'HybridConvLSTMV2':  HybridConvLSTMModelV2,
    'HybridConvLSTMV3': HybridConvLSTMV3,
    'HybridConvLSTMBidirectional': HybridConvLSTMBidirectionalModel,
    'AttentionModel': AttentionModel,
    'HybridConvAttentionGRUModel': HybridConvAttentionGRUModel,
    'Seq2SeqLSTMV1': Seq2SeqTrajectoryLSTMV1,
    'Seq2SeqLSTMV2': Seq2SeqTrajectoryLSTMV2
}

def run_experiment(model_name, hidden_size=64, num_layers=2,
                   task='next_instance'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if task == 'next_instance': 
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(input_size=21, hidden_size=hidden_size,
                            num_layers=num_layers, output_size=2).to(device)
    elif task == 'next_ten_mins':
        model_class = MODEL_REGISTRY[model_name]
        model = model_class(input_size=21, hidden_size=hidden_size,
                            num_layers=num_layers, output_size=20).to(device)
    
    return model

