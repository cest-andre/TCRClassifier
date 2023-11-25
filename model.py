import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Install (huggingface) transformers package by pip:
# pip install transformers
from transformers import BertModel, BertConfig  
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

BERT_CONFIG = BertConfig(
    vocab_size=25,  
    max_position_embeddings=64,
    type_vocab_size=2,
    num_attention_heads=8,
    num_hidden_layers=8,
    hidden_size=512,
    intermediate_size=2048,
    num_labels=2
)


class FocalLoss(nn.Module):
    """
    Focal loss implementation as nn.Module
    """
    def __init__(
        self,
        gamma: float = 0,
        alpha: float = None,
        size_average: bool = True,
        no_agg: bool = False,
    ):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.size_average = size_average
        self.no_agg = no_agg

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes focal loss fo a batch
        :param input: model predictions (two values for each data point, no softmax)
        :param target: true class labels (0 or 1)
        :return: loss for every observation if `no_agg` is True, otherwise
            average loss if `size_average` is True, else sum of losses
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.no_agg:
            return loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


# loss function to be used for classification training
clf_loss_func = FocalLoss(gamma=3, alpha=0.25, no_agg=True)    


class TCRModel(BertModel):
    def __init__(self):
        super().__init__(BERT_CONFIG)  # Transformer
        self.classifier = RobertaClassificationHead(BERT_CONFIG)  # Dense net for classification, output without softmax
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        classification: bool = True
    ):
        """
        :param input_ids: amino acid index numbers (
        :param attention_mask: attention mask (1 for non-padding token and 0 for padding)
        :param classification: flag whether to perform classification from Transformer output at position 0 
        """
        transformer_outputs = super().forward(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        sequence_output = transformer_outputs[0]
        
        if classification:
            return self.classifier(sequence_output)
        else:
            return sequence_output


    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn, map_location=None):
        self.load_state_dict(torch.load(fn, map_location))
    