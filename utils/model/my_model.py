import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from utils.model.examplar import ExemplarHandler


class Attention(nn.Module):
    ''' Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = Attention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        k = torch.cat([q, k], 1)
        v = torch.cat([q, v], 1)
        len_k = len_k + len_q
        len_v = len_v + len_q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = output + residual
        return output


class my_model(ExemplarHandler):
    def __init__(self, backbone=None, pretrained=False, args=None):
        super(my_model, self).__init__()
        self.backbone = backbone(pretrained=pretrained)
        self.fc_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding = nn.Sequential(
            nn.Conv2d(self.fc_features * 2, args.embedding, 1),
            nn.BatchNorm2d(args.embedding),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(args.embedding, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(64, 1, 1),
        )
        self.args = args
        self.proto_all = nn.ParameterList([])
        self.IL_attn = MultiHeadAttention(1, 512, self.args.latent_dim, self.args.latent_dim, dropout=0.1)
        self.add_classes(self.args.base_class)

    def forward(self, query_image, support_image, support_target):
        support_embedding = self.backbone(support_image)
        support_labels, support_idx = self.selected_batch(support_target)
        # shape (3,5,5) (3,5)
        support_embedding = support_embedding[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
        support_embedding = support_embedding.mean(dim=2)
        outputs = []
        for tt in range(self.args.batch_task):
            base_mask = torch.eye(self.args.base_class).cuda()
            support_index = support_labels[tt, :]
            base_mask[support_index, :] = 0
            # 60*5
            incremental_mask = self.project(support_index, self.args.base_class)
            classifier_old = torch.mm(base_mask, self.proto) + torch.mm(incremental_mask, support_embedding[tt, :])
            proto_old = torch.mm(base_mask, self.proto)
            classifier_new = self.IL_attn(classifier_old.unsqueeze(0), proto_old.unsqueeze(0), proto_old.unsqueeze(0))
            features = self.backbone(query_image)

            output = self.forward_metric(features, classifier_new[0])
            outputs.append(output)
        outputs = torch.cat(outputs, 1)
        outputs = outputs.view(-1, self.args.base_class)
        return outputs

    def forward_test(self, inputs):
        features = self.backbone(inputs)
        output = self.forward_metric(features, self.proto_all[-1])
        return output

    def forward_metric(self, features, proto):
        batch_size = features.shape[0]
        num_class = proto.shape[0]
        features = F.normalize(features, p=2, dim=1)
        features = features.view((batch_size, 1, -1, 1, 1))
        features = features.repeat((1, num_class, 1, 1, 1))
        proto = F.normalize(proto, p=2, dim=1)
        proto = proto.view((1, num_class, self.fc_features, 1, 1))
        proto = proto.repeat((batch_size, 1, 1, 1, 1))
        features = torch.cat([features, proto], dim=2)
        features = features.view((-1, 2 * self.fc_features, 1, 1))
        output = self.embedding(features)
        output = output.view((batch_size, num_class))
        return output

    def feature_extractor(self, images):
        features = self.backbone(images)
        batch, channel, _, _ = features.size()
        features = features.view(batch, channel)
        return features

    # cub no memory
    def calculate_means_cub1(self, image):
        embedding = self.backbone(image).reshape((-1, self.args.shot, self.fc_features))
        embedding = embedding.mean(dim=1)
        return embedding

    def calculate_means_cub2(self, embeddings):
        classifier_old = torch.cat([self.proto_all[0], embeddings])
        proto_old = self.proto_all[0]
        classifier_new = self.IL_attn(classifier_old.unsqueeze(0), proto_old.unsqueeze(0), proto_old.unsqueeze(0))
        classifier_current = torch.cat([self.proto_all[0], classifier_new[0][self.args.base_class:]], dim=0)
        self.proto_all.append(nn.Parameter(classifier_current))

    def calculate_means(self, image):
        embedding = self.backbone(image).reshape((-1, self.args.shot, self.fc_features))
        embedding = embedding.mean(dim=1)
        classifier_old = torch.cat([self.proto_all[0], embedding])
        proto_old = self.proto_all[0]
        classifier_new = self.IL_attn(classifier_old.unsqueeze(0), proto_old.unsqueeze(0), proto_old.unsqueeze(0))
        self.proto_all.append(nn.Parameter(classifier_new[0]))

    @property
    def proto(self):
        return self.proto_all[-1]

    @property
    def new_proto(self):
        return self.proto_all[-1]

    def add_classes(self, n_classes):
        self.proto_all.append(nn.Parameter(torch.zeros(n_classes, self.fc_features)))
        self.cuda()

    def selected_batch(self, support_label):
        args = self.args
        selected_ids = []
        selected_way = []
        support_label = support_label.view(args.way+3, args.shot)
        for i in range(self.args.batch_task):
            selected_class = torch.randperm(args.way+3)[:args.way]
            selected_way.append(support_label[selected_class, 0])
            selected_id = selected_class.view(args.way, 1) * args.shot + torch.arange(args.shot).view(1, args.shot)
            selected_ids.append(selected_id)
        selected_way, selected_ids = torch.stack(selected_way), torch.stack(selected_ids)
        return selected_way, selected_ids

    def project(self, support_index, num_class):

        encoded_index = torch.zeros(torch.Size([num_class]) + support_index.size())
        if support_index.is_cuda:
            encoded_index = encoded_index.cuda()
        index = support_index.view(torch.Size([1]) + support_index.size())
        encoded_index = encoded_index.scatter_(0, index, 1)

        return encoded_index

