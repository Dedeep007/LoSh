"""
Modified from DETR https://github.com/facebookresearch/detr
"""

import torch
import torch.nn.functional as F
from torch import nn
from misc import NestedTensor
from models.backbone import init_backbone
from models.matcher import build_matcher
from models.segmentation import FPNSpatialDecoder
from models.multimodal_transformer import MultimodalTransformer
from models.criterion import SetCriterion
from models.postprocessing import A2DSentencesPostProcess, ReferYoutubeVOSPostProcess
from einops import rearrange, repeat


class MTTR(nn.Module):
    """ The main module of the Multimodal Tracking Transformer """
    def __init__(self, num_queries, mask_kernels_dim=8, aux_loss=False, **kwargs):
        """
        Parameters:
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         MTTR can detect in a single image. In our paper we use 50 in all settings.
            mask_kernels_dim: dim of the segmentation kernels and of the feature maps outputted by the spatial decoder.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = init_backbone(**kwargs)
        self.transformer = MultimodalTransformer(**kwargs)
        d_model = self.transformer.d_model
        self.is_referred_head = nn.Linear(d_model, 2)  # binary 'is referred?' prediction head for object queries
        self.instance_kernels_head = MLP(d_model, d_model, output_dim=mask_kernels_dim, num_layers=2)
        self.obj_queries = nn.Embedding(num_queries, d_model)  # pos embeddings for the object queries
        self.vid_embed_proj = nn.Conv2d(self.backbone.layer_output_channels[-1], d_model, kernel_size=1)
        self.spatial_decoder = FPNSpatialDecoder(d_model, self.backbone.layer_output_channels[:-1][::-1], mask_kernels_dim)
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor, valid_indices, text_queries, short_text_queries):
        """The forward expects a NestedTensor, which consists of:
               - samples.tensor: Batched frames of shape [time x batch_size x 3 x H x W]
               - samples.mask: A binary mask of shape [time x batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_is_referred": The reference prediction logits for all queries.
                                     Shape: [time x batch_size x num_queries x 2]
               - "pred_masks": The mask logits for all queries.
                               Shape: [time x batch_size x num_queries x H_mask x W_mask]
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        backbone_out = self.backbone(samples)
        # keep only the valid frames (frames which are annotated):
        # (for example, in a2d-sentences only the center frame in each window is annotated).
        # print("before: ",len(backbone_out))
        # for i in range(len(backbone_out)):
        #   print("NestedTensor")
        #   print(backbone_out[i].tensors.shape)
        #   print(backbone_out[i].mask.shape)
        for layer_out in backbone_out:
            layer_out.tensors = layer_out.tensors.index_select(0, valid_indices)
            layer_out.mask = layer_out.mask.index_select(0, valid_indices)
        # print("after: ",len(backbone_out))
        # for i in range(len(backbone_out)):
        #   print("NestedTensor")
        #   print(backbone_out[i].tensors.shape)
        #   print(backbone_out[i].mask.shape)
        bbone_final_layer_output = backbone_out[-1]
        # print("backbone_out[-1].tensors.shape: ", bbone_final_layer_output.tensors.shape)
        # print("backbone_out[-1].mask.shape: ", bbone_final_layer_output.mask.shape)
        vid_embeds, vid_pad_mask = bbone_final_layer_output.decompose()
        # print("vid_embeds.shape: ",vid_embeds.shape)
        # print("vid_pad_mask.shape: ", vid_pad_mask.shape)

        T, B, _, _, _ = vid_embeds.shape
        vid_embeds = rearrange(vid_embeds, 't b c h w -> (t b) c h w')
        vid_embeds = self.vid_embed_proj(vid_embeds)
        vid_embeds = rearrange(vid_embeds, '(t b) c h w -> t b c h w', t=T, b=B)

        transformer_out = self.transformer(vid_embeds, vid_pad_mask, text_queries, short_text_queries, self.obj_queries.weight)
        # hs is: [L, T, B, N, D] where L is number of decoder layers
        # vid_memory is: [T, B, D, H, W]
        # txt_memory is a list of length T*B of [S, C] where S might be different for each sentence
        # encoder_middle_layer_outputs is a list of [T, B, H, W, D]
        long_hs, long_vid_memory, long_txt_memory, short_hs, short_vid_memory, short_txt_memory, = transformer_out
        # print("hs.shape[L,T,B,N,D]: ", hs.shape) #[3,1,3,50,256]

        # hs = hs_long cat hs_short
        hs = torch.cat([long_hs,short_hs], dim=2)
        vid_memory = torch.cat([long_vid_memory, short_vid_memory], dim=1)

        vid_memory = rearrange(vid_memory, 't b d h w -> (t b) d h w')
        #bbone_middle_layer_outputs = [rearrange(o.tensors, 't b d h w -> (t b) d h w') for o in backbone_out[:-1][::-1]]
        bbone_middle_layer_outputs = [repeat(rearrange(o.tensors, 't b d h w -> (t b) d h w'), 't_b d h w -> (t t_b) d h w', t=2) for o in backbone_out[:-1][::-1]]

        decoded_frame_features = self.spatial_decoder(vid_memory, bbone_middle_layer_outputs)
        decoded_frame_features = rearrange(decoded_frame_features, '(t b) d h w -> t b d h w', t=T, b=2*B)
        instance_kernels = self.instance_kernels_head(hs)  # [L, T, B, N, C]
        # output masks is: [L, T, B, N, H_mask, W_mask]
        output_masks = torch.einsum('ltbnc,tbchw->ltbnhw', instance_kernels, decoded_frame_features)
        outputs_is_referred = self.is_referred_head(hs)  # [L, T, B, N, 2]


        #print("before:", output_masks.shape)
        long_output_masks = output_masks[:,:,:B,:,:]
        #print("after:", long_output_masks.shape)
        long_outputs_is_referred = outputs_is_referred[:,:,:B,:,:]

        short_output_masks = output_masks[:,:,B:,:,:]
        short_outputs_is_referred = outputs_is_referred[:,:,B:,:,:]


        layer_outputs = []
        for pm, pir in zip(long_output_masks, long_outputs_is_referred):
            layer_out = {'pred_masks': pm, # T B N H W
                         'pred_is_referred': pir} # T B N 2
            layer_outputs.append(layer_out)
        long_out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        # print("self.aux_loss: ",self.aux_loss) # True by default
        if self.aux_loss:
            long_out['aux_outputs'] = layer_outputs[:-1]


        layer_outputs = []
        for pm, pir in zip(short_output_masks, short_outputs_is_referred):
            layer_out = {'pred_masks': pm,
                         'pred_is_referred': pir}
            layer_outputs.append(layer_out)
        short_out = layer_outputs[-1]  # the output for the last decoder layer is used by default
        # print("self.aux_loss: ",self.aux_loss) # True by default


        if self.aux_loss:
            short_out['aux_outputs'] = layer_outputs[:-1]

            
        return [long_out, short_out]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = args.device
    model = MTTR(**vars(args))
    matcher = build_matcher(args)
    weight_dict = {'loss_is_referred': args.is_referred_loss_coef,
                   'loss_dice': args.dice_loss_coef,
                   'loss_sigmoid_focal': args.sigmoid_focal_loss_coef,
                   'loss_conditioned_iou': args.conditioned_iou_loss_coef}
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict, eos_coef=args.eos_coef)
    criterion.to(device)
    if args.dataset_name == 'a2d_sentences' or args.dataset_name == 'jhmdb_sentences':
        postprocessor = A2DSentencesPostProcess()
    elif args.dataset_name == 'ref_youtube_vos':
        postprocessor = ReferYoutubeVOSPostProcess()
    else:
        assert False, f'postprocessing for dataset: {args.dataset_name} is not supported'
    return model, criterion, postprocessor
