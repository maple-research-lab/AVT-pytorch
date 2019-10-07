import torch
import torch.nn as nn
import imp
import os


current_path = os.path.abspath(__file__)
filepath_to_linear_classifier_definition = os.path.join(os.path.dirname(current_path), 'LinearClassifier.py')
LinearClassifier = imp.load_source('',filepath_to_linear_classifier_definition).create_model


class MClassifier(nn.Module):
    def __init__(self, opts):
        super(MClassifier, self).__init__()
        self.classifiers = nn.ModuleList([LinearClassifier(opt) for opt in opts])
        self.num_classifiers = len(opts)


    def forward(self, feats):
        assert(len(feats) == self.num_classifiers)
        outputs = []
        for i, feat in enumerate(feats):
            if i == 3:
                out_mean = feat[:, :256]
                out_var = feat[:, 256:]
                std = torch.exp(0.5*out_var)
                feats = []
                for _ in range(5):
                    eps = torch.randn_like(std)
                    feat = eps.mul(std * 0.001).add_(out_mean)
                    feats.append(feat)
                
                feat = feats[0]
                for ii in range(1, 5):
                    feat += feats[ii]
                
                feat = feat / 5.
            
            outputs = outputs + [self.classifiers[i](feat)]
        return outputs

def create_model(opt):
    return MClassifier(opt)

