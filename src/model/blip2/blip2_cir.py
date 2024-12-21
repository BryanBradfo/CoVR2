# """
# Copyright (c) 2023, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# """

# import logging
# from typing import Any

# import torch
# import torch.nn as nn
# from torch.cuda.amp import autocast as autocast
# from torch.nn import functional as F

# from src.model.blip2.blip2 import Blip2Base, disabled_train
# from src.tools.utils import all_gather_with_grad, concat_all_gather
# from src.model.blip2.Qformer import BertModel

# from transformers.models.bert.configuration_bert import BertConfig


# class BLIP2Cir(Blip2Base):
#     """
#     BLIP2 first-stage model with Q-former and ViT.
#     Supported model types:
#         - pretrained: pretrained model with vit-g
#         - pretrain_vitL: pretrained model with vit-large
#         - coco: fintuned model on coco
#     Usage:
#         >>> from lavis.models import load_model
#         >>> model = load_model("blip2", "pretrain")
#     """

#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#     }

#     def __init__(
#         self,
#         loss: Any,
#         vit_model="clip_L",
#         image_size=224,
#         drop_path_rate=0,
#         use_grad_checkpoint=False,
#         vit_precision="fp32",
#         train_vit=False,
#         vit="large",
#         num_query_token=32,
#         cross_attention_freq=2,
#         embed_dim=256,
#         max_txt_len=32,
#         temperature=1,
#         si_ti_weight=1,
#         si_tc_weight=0,

#         med_config="configs/med_config.json",
#     ):
#         super().__init__()

#         self.loss = loss

#         self.tokenizer = self.init_tokenizer()

#         self.visual_encoder, self.ln_vision, vision_width = self.init_vision_encoder(
#             vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
#         )
#         self.train_vit = train_vit
#         if not train_vit:
#             for name, param in self.visual_encoder.named_parameters():
#                 param.requires_grad = False
#             self.visual_encoder = self.visual_encoder.eval()
#             self.visual_encoder.train = disabled_train
#             logging.info("freeze vision encoder")
#         self.Qformer, self.query_tokens = self.init_Qformer(
#             num_query_token, self.visual_encoder.num_features, cross_attention_freq
#         )
#         med_config = BertConfig.from_json_file(med_config)
#         med_config.encoder_width = vision_width
#         self.text_encoder_only = BertModel(config=med_config, add_pooling_layer=False)
#         self.Qformer.resize_token_embeddings(len(self.tokenizer))
#         state_dict = self.Qformer.state_dict()
#         for name, param in self.Qformer.named_parameters():
#             if "_query" in name:
#                 key_orig = name.replace("_query", "")
#                 param.data.copy_(state_dict[key_orig])

#         self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
#         self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
#         self.text_only_proj = nn.Linear(embed_dim*3, embed_dim)

#         self.temp = temperature
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * 3, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3),
#             nn.Softmax(dim=1),
#         )
#         self.training_type = "txt_embs_out"

#         self.max_txt_len = max_txt_len

#         for p in self.vision_proj.parameters():
#             p.requires_grad = False

#         for p in self.ln_vision.parameters():
#             p.requires_grad = False

#         for p in self.Qformer.cls.parameters():
#             p.requires_grad = False

#         assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
#         self.si_ti_weight = si_ti_weight
#         self.si_tc_weight = si_tc_weight

#     def forward(self, batch, fabric):
#         ref_img = batch["ref_img"]
#         tar_img_feat = batch["tar_img_feat"]
#         caption = batch["edit"]

#         ref_img.half()

#         device = ref_img.device

#         # Encode the target image
#         tar_img_feat = tar_img_feat.to(device)
#         tar_img_feat = concat_all_gather(tar_img_feat, fabric)

#         # Text
#         text_tokens = self.tokenizer(
#             caption,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_txt_len,
#             return_tensors="pt",
#         ).to(device)

#         if self.train_vit:
#             ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
#         else:
#             with torch.no_grad():
#                 ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

#         # Encode the reference image
#         ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

#         ###============== Image-text Matching ===================###
#         query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
#         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
#             self.device
#         )
#         attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

#         output = self.Qformer.bert(
#             text_tokens.input_ids,  # [bs, 32]
#             query_embeds=query_tokens,  # [bs, 32, 768]
#             attention_mask=attention_mask,  # [bs, 64]
#             encoder_hidden_states=ref_img_embs,  # [bs, 677, 1408]
#             encoder_attention_mask=ref_img_atts,  # [bs, 677]
#             return_dict=True,
#         )

#         encoder_input_ids = text_tokens.input_ids.clone()
#         text_feat = self.text_encoder_only(
#             encoder_input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#             mode="text",
#         )

#         text_feat = text_feat.last_hidden_state[:, 0, :]
#         text_feat = F.normalize(self.text_proj(text_feat), dim=-1)


#         vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
#         query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
#         query_si_feat = all_gather_with_grad(query_si_feat, fabric)

#         # mean over all query tokens
#         query_si_feat = query_si_feat.mean(dim=1)
#         tar_img_feat = tar_img_feat.mean(dim=1)

#         img_feat_2d = F.normalize(self.vision_proj(ref_img_embs.mean(dim=1)), dim=-1)
#         concatenated_feats = torch.cat(
#             (query_si_feat.unsqueeze(1), img_feat_2d.unsqueeze(1), text_feat.unsqueeze(1)),
#             dim=1,
#         )
#         combined_query_feat = concatenated_feats.view(concatenated_feats.size(0), -1)
        
#         weights = self.mlp(combined_query_feat)
#         query_si_feat = (
#             weights[:, 0].unsqueeze(1) * query_si_feat
#             + weights[:, 1].unsqueeze(1) * img_feat_2d
#             + weights[:, 2].unsqueeze(1) * text_feat
#         )
#         # s=source, t=target, i=image, c=caption, w=weight
#         loss = 0
#         if self.si_ti_weight > 0:
#             si_ti_loss = self.loss(query_si_feat, tar_img_feat, self.temp)
#             loss += si_ti_loss * self.si_ti_weight

#         if self.si_tc_weight > 0:
#             assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
#             tar_txt_feat = batch["tar_txt_feat"]

#             tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)

#             si_tc_loss = self.loss(query_si_feat, tar_txt_feat, self.temp)
#             loss += si_tc_loss * self.si_tc_weight

#         return loss


# def blip2_cir(model, ckpt_path, **kwargs):
#     if ckpt_path:
#         model.load_from_pretrained(url_or_filename=ckpt_path)
#     return model

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather
# On importe le BertConfig personnalisé ou on peut utiliser celui d'HuggingFace,
# mais on doit alors lui ajouter les attributs par set/getattr.
from src.model.blip2.Qformer import BertModel, BertConfig


class BLIP2Cir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="clip_L",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,  # <--- on récupère ce param
        embed_dim=256,
        max_txt_len=32,
        temperature=1,
        si_ti_weight=1,
        si_tc_weight=0,
        med_config="/home/bryanbradfo/CoVR2/configs/med_config.json",
    ):
        super().__init__()

        self.loss = loss
        self.tokenizer = self.init_tokenizer()

        # --------------------------------------------------------------------
        # 1) Initialisation de l’encodeur visuel
        # --------------------------------------------------------------------
        (
            self.visual_encoder,
            self.ln_vision,
            vision_width,
        ) = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )

        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # --------------------------------------------------------------------
        # 2) Initialisation de la Q-former
        # --------------------------------------------------------------------
        # On crée la config Bert à partir du fichier JSON
        med_config = BertConfig.from_json_file(med_config)
        # On lui injecte les attributs manquants
        med_config.cross_attention_freq = cross_attention_freq
        med_config.add_cross_attention = True
        med_config.encoder_width = vision_width

        # On crée ensuite la Q-former + query_tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token,
            self.visual_encoder.num_features,  # ex: 1408 pour ViT-L
            cross_attention_freq,
        )

        # --------------------------------------------------------------------
        # 3) Initialise "text_encoder_only" (un BERTModel custom)
        # --------------------------------------------------------------------
        self.text_encoder_only = BertModel(config=med_config, add_pooling_layer=False)

        # Ajuste les embeddings pour le Q-former
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:  # Copie des poids _query -> query
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # --------------------------------------------------------------------
        # 4) Projections linéaires pour vision et texte
        # --------------------------------------------------------------------
        # Dans certains BLIP2, vision_width = 1408 (ViT-L). On projette en embed_dim.
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        # Q-former a un hidden_size = 768 -> projection en embed_dim
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_only_proj = nn.Linear(embed_dim * 3, embed_dim)

        self.temp = temperature
        self.max_txt_len = max_txt_len

        # MLP de fusion
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=1),
        )

        self.training_type = "txt_embs_out"

        # Option : freezers
        for p in self.vision_proj.parameters():
            p.requires_grad = False
        for p in self.ln_vision.parameters():
            p.requires_grad = False
        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

    def forward(self, batch, fabric):
        ref_img = batch["ref_img"]
        tar_img_feat = batch["tar_img_feat"]
        caption = batch["edit"]

        # On met l’image en FP16 si nécessaire
        ref_img = ref_img.half()
        device = ref_img.device

        # On rassemble les features cibles (déjà extraites)
        tar_img_feat = tar_img_feat.to(device)
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)

        # Tokenize texte
        text_tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)

        # ----- Encodeur visuel sur l’image de référence -----
        if self.train_vit:
            ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
        else:
            with torch.no_grad():
                ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

        ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

        # ----- Q-former : Image + texte -----
        query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

        output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=ref_img_embs,
            encoder_attention_mask=ref_img_atts,
            return_dict=True,
        )

        # ----- Petit BERT encodeur pour le texte seulement -----
        text_feat_out = self.text_encoder_only(
            text_tokens.input_ids.clone(),
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
            mode="text",  # si c’est géré dans votre code
        )
        # On récupère la sortie [CLS] (ou le 1er token) et on le normalise
        text_feat = text_feat_out.last_hidden_state[:, 0, :]
        text_feat = F.normalize(self.text_proj(text_feat), dim=-1)

        # ----- Feature Q-former (mix image + texte) -----
        vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
        query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
        query_si_feat = all_gather_with_grad(query_si_feat, fabric)
        query_si_feat = query_si_feat.mean(dim=1)

        # ----- Feature pure image (référence) -----
        img_feat_2d = ref_img_embs.mean(dim=1)
        img_feat_2d = F.normalize(self.vision_proj(img_feat_2d), dim=-1)

        # ----- Fusion par MLP -----
        concatenated_feats = torch.cat(
            (
                query_si_feat.unsqueeze(1),
                img_feat_2d.unsqueeze(1),
                text_feat.unsqueeze(1),
            ),
            dim=1,
        )  # [bs, 3, embed_dim]
        combined_query_feat = concatenated_feats.view(concatenated_feats.size(0), -1)
        weights = self.mlp(combined_query_feat)  # => [bs, 3]
        fused_query_si_feat = (
            weights[:, 0].unsqueeze(1) * query_si_feat
            + weights[:, 1].unsqueeze(1) * img_feat_2d
            + weights[:, 2].unsqueeze(1) * text_feat
        )

        # ----- Calcul des pertes -----
        loss = 0
        if self.si_ti_weight > 0:
            tar_img_feat = tar_img_feat.mean(dim=1)
            si_ti_loss = self.loss(fused_query_si_feat, tar_img_feat, self.temp)
            loss += si_ti_loss * self.si_ti_weight

        if self.si_tc_weight > 0:
            assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
            tar_txt_feat = batch["tar_txt_feat"]
            tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)
            si_tc_loss = self.loss(fused_query_si_feat, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight

        return loss


def blip2_cir(model, ckpt_path, **kwargs):
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model










# import logging
# from typing import Any

# import torch
# import torch.nn as nn
# from torch.cuda.amp import autocast as autocast
# from torch.nn import functional as F

# from src.model.blip2.blip2 import Blip2Base, disabled_train
# from src.tools.utils import all_gather_with_grad, concat_all_gather
# from src.model.blip2.Qformer import BertModel, BertConfig

# # from transformers.models.bert.configuration_bert import BertConfig


# class BLIP2Cir(Blip2Base):
#     """
#     BLIP2 first-stage model with Q-former and ViT.
#     Supported model types:
#         - pretrained: pretrained model with vit-g
#         - pretrain_vitL: pretrained model with vit-large
#         - coco: fintuned model on coco
#     Usage:
#         >>> from lavis.models import load_model
#         >>> model = load_model("blip2", "pretrain")
#     """

#     PRETRAINED_MODEL_CONFIG_DICT = {
#         "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
#         "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
#         "coco": "configs/models/blip2/blip2_coco.yaml",
#     }

#     def __init__(
#         self,
#         loss: Any,
#         vit_model="clip_L",
#         image_size=224,
#         drop_path_rate=0,
#         use_grad_checkpoint=False,
#         vit_precision="fp32",
#         train_vit=False,
#         vit="large",
#         num_query_token=32,
#         cross_attention_freq=2,
#         embed_dim=256,
#         max_txt_len=32,
#         temperature=1,
#         si_ti_weight=1,
#         si_tc_weight=0,
#         med_config="/home/bryanbradfo/CoVR2/configs/med_config.json",
#     ):
#         super().__init__()

#         self.loss = loss
#         self.tokenizer = self.init_tokenizer()

#         # --------------------------------------------------------------------
#         # 1) Initialisation de l’encodeur visuel
#         # --------------------------------------------------------------------
#         (
#             self.visual_encoder,
#             self.ln_vision,
#             vision_width,
#         ) = self.init_vision_encoder(
#             vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
#         )

#         self.train_vit = train_vit
#         if not train_vit:
#             for name, param in self.visual_encoder.named_parameters():
#                 param.requires_grad = False
#             self.visual_encoder = self.visual_encoder.eval()
#             self.visual_encoder.train = disabled_train
#             logging.info("freeze vision encoder")

#         # --------------------------------------------------------------------
#         # 2) Initialisation de la Q-former
#         # --------------------------------------------------------------------
#         self.Qformer, self.query_tokens = self.init_Qformer(
#             num_query_token,
#             self.visual_encoder.num_features,  # ex : 1408 pour Vit-L
#             cross_attention_freq,
#         )

#         # --------------------------------------------------------------------
#         # 3) Initialisation du petit encodeur BERT ("text_encoder_only")
#         # --------------------------------------------------------------------
#         med_config = BertConfig.from_json_file(med_config)
#         med_config.encoder_width = vision_width
#         self.text_encoder_only = BertModel(config=med_config, add_pooling_layer=False)

#         # Ajuste les embeddings pour le Q-former
#         self.Qformer.resize_token_embeddings(len(self.tokenizer))
#         state_dict = self.Qformer.state_dict()
#         for name, param in self.Qformer.named_parameters():
#             # Copie des poids _query vers query
#             if "_query" in name:
#                 key_orig = name.replace("_query", "")
#                 param.data.copy_(state_dict[key_orig])

#         # --------------------------------------------------------------------
#         # 4) Projections linéaires pour vision et texte
#         # --------------------------------------------------------------------
#         # IMPORTANT : Pour projeter directement la sortie du vision encoder
#         # qui est (batch_size, nb_patches, vision_width), il faut mettre
#         # vision_width en dimension d’entrée de la projection.
#         # Par exemple, vision_width = 1408 pour la ViT-L.
#         self.vision_proj = nn.Linear(vision_width, embed_dim)

#         # Pour projeter les sorties textuelles (ou Q-former),
#         # la dimension d’entrée est la hidden_size du Q-former (768 en général).
#         self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

#         # Optionnel : si vous souhaitez faire une fusion plus complexe
#         self.text_only_proj = nn.Linear(embed_dim * 3, embed_dim)

#         # --------------------------------------------------------------------
#         # 5) Paramètres supplémentaires
#         # --------------------------------------------------------------------
#         self.temp = temperature
#         self.max_txt_len = max_txt_len

#         # MLP qui génère un poids (via softmax) pour chaque embedding
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim * 3, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 3),
#             nn.Softmax(dim=1),
#         )
#         self.training_type = "txt_embs_out"

#         # On peut éventuellement freezer la vision_proj si on ne veut pas l’entraîner
#         for p in self.vision_proj.parameters():
#             p.requires_grad = False

#         for p in self.ln_vision.parameters():
#             p.requires_grad = False

#         # On peut freezer ou non les poids du CLS Qformer
#         for p in self.Qformer.cls.parameters():
#             p.requires_grad = False

#         assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
#         self.si_ti_weight = si_ti_weight
#         self.si_tc_weight = si_tc_weight

#     def forward(self, batch, fabric):
#         ref_img = batch["ref_img"]
#         tar_img_feat = batch["tar_img_feat"]
#         caption = batch["edit"]

#         # Par sécurité : si l’image est en FP16
#         ref_img = ref_img.half()
#         device = ref_img.device

#         # Encode la target image (features déjà extraits, on les rassemble)
#         tar_img_feat = tar_img_feat.to(device)
#         tar_img_feat = concat_all_gather(tar_img_feat, fabric)

#         # Tokenize le texte
#         text_tokens = self.tokenizer(
#             caption,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_txt_len,
#             return_tensors="pt",
#         ).to(device)

#         # --------------------------------------------------------------------
#         # 1) Encodeur visuel sur l’image de référence
#         # --------------------------------------------------------------------
#         if self.train_vit:
#             ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))
#         else:
#             with torch.no_grad():
#                 ref_img_embs = self.ln_vision(self.visual_encoder(ref_img))

#         ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

#         # --------------------------------------------------------------------
#         # 2) Q-former : Image + texte
#         # --------------------------------------------------------------------
#         query_tokens = self.query_tokens.expand(ref_img_embs.shape[0], -1, -1)
#         query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)

#         attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)

#         output = self.Qformer.bert(
#             text_tokens.input_ids,         # [bs, max_txt_len]
#             query_embeds=query_tokens,     # [bs, num_query_token, Qformer_dim]
#             attention_mask=attention_mask, # [bs, num_query_token + max_txt_len]
#             encoder_hidden_states=ref_img_embs,  # [bs, n_patches, vision_width]
#             encoder_attention_mask=ref_img_atts, # [bs, n_patches]
#             return_dict=True,
#         )

#         # --------------------------------------------------------------------
#         # 3) Encodeur texte "classique" (BERT)
#         # --------------------------------------------------------------------
#         encoder_input_ids = text_tokens.input_ids.clone()
#         text_feat_out = self.text_encoder_only(
#             encoder_input_ids,
#             attention_mask=text_tokens.attention_mask,
#             return_dict=True,
#             mode="text",
#         )
#         # on récupère le [CLS] ou bien le premier token
#         text_feat = text_feat_out.last_hidden_state[:, 0, :]
#         # projection
#         text_feat = F.normalize(self.text_proj(text_feat), dim=-1)

#         # --------------------------------------------------------------------
#         # 4) Récupération de la feature Q-former (image + texte)
#         # --------------------------------------------------------------------
#         vl_embs = output.last_hidden_state[:, : query_tokens.size(1), :]
#         # vl_embs = [bs, num_query_tokens, Qformer_dim]
#         query_si_feat = F.normalize(self.text_proj(vl_embs), dim=-1)
#         query_si_feat = all_gather_with_grad(query_si_feat, fabric)
#         # moyenne sur tous les tokens de la Q-former
#         query_si_feat = query_si_feat.mean(dim=1)

#         # --------------------------------------------------------------------
#         # 5) Récupération (et normalisation) de la feature image brute
#         # --------------------------------------------------------------------
#         img_feat_2d = ref_img_embs.mean(dim=1)  # [bs, vision_width]
#         img_feat_2d = F.normalize(self.vision_proj(img_feat_2d), dim=-1)

#         # --------------------------------------------------------------------
#         # 6) Fusion via MLP (weights)
#         # --------------------------------------------------------------------
#         # On concatène (query_si_feat, img_feat_2d, text_feat)
#         # chacun étant [bs, embed_dim] -> on obtient [bs, 3, embed_dim]
#         concatenated_feats = torch.cat(
#             (
#                 query_si_feat.unsqueeze(1),
#                 img_feat_2d.unsqueeze(1),
#                 text_feat.unsqueeze(1),
#             ),
#             dim=1,
#         )  # [bs, 3, embed_dim]

#         # Flatten en [bs, 3*embed_dim]
#         combined_query_feat = concatenated_feats.view(concatenated_feats.size(0), -1)

#         # On passe dans un MLP qui sort 3 poids (Softmax)
#         weights = self.mlp(combined_query_feat)  # [bs, 3]

#         # Combinaison pondérée
#         fused_query_si_feat = (
#             weights[:, 0].unsqueeze(1) * query_si_feat
#             + weights[:, 1].unsqueeze(1) * img_feat_2d
#             + weights[:, 2].unsqueeze(1) * text_feat
#         )

#         # --------------------------------------------------------------------
#         # 7) Calcul des différentes losses
#         # --------------------------------------------------------------------
#         loss = 0
#         if self.si_ti_weight > 0:
#             # Similarité source-image vs target-image
#             tar_img_feat = tar_img_feat.mean(dim=1)
#             si_ti_loss = self.loss(fused_query_si_feat, tar_img_feat, self.temp)
#             loss += si_ti_loss * self.si_ti_weight

#         if self.si_tc_weight > 0:
#             # Similarité source-image vs target-text
#             assert "tar_txt_feat" in batch, "tar_txt_feat is not in batch"
#             tar_txt_feat = batch["tar_txt_feat"]
#             tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)
#             si_tc_loss = self.loss(fused_query_si_feat, tar_txt_feat, self.temp)
#             loss += si_tc_loss * self.si_tc_weight

#         return loss


# def blip2_cir(model, ckpt_path, **kwargs):
#     if ckpt_path:
#         model.load_from_pretrained(url_or_filename=ckpt_path)
#     return model

