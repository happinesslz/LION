import torch
from torch.nn import functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its
    detector forward function and use learnable tgt embedding, so we change this function a little bit.
    :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn
    :return:
    """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args

        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["gt_boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # only apply on x and y axis
            known_bbox_[:, :3] = known_bboxs[:, :3] - known_bboxs[:, 3:6] / 2
            known_bbox_[:, 3:6] = known_bboxs[:, :3] + known_bboxs[:, 3:6] / 2
            known_bbox_[:, 6:] = known_bboxs[:, 6:]

            diff = torch.zeros_like(known_bboxs)
            diff[:, :3] = known_bboxs[:, 3:6] / 2
            diff[:, 3:6] = known_bboxs[:, 3:6] / 2
            diff[:, 6:] = 0.1  # 36 degree

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign

            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :3] = (known_bbox_[:, :3] + known_bbox_[:, 3:6]) / 2
            known_bbox_expand[:, 3:6] = known_bbox_[:, 3:6] - known_bbox_[:, :3]
            known_bbox_expand[:, 6:] = known_bbox_[:, 6:]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed_v2 = label_enc(m) 
        input_label_embed = F.one_hot(m, num_classes=num_classes).float()
        input_bbox_embed = known_bbox_expand

        padding_label = torch.zeros(pad_size, num_classes).cuda()
        padding_label_v2 = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 7).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_label_v2 = padding_label_v2.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_label_v2[(known_bid.long(), map_known_indice)] = input_label_embed_v2
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct GTs
        attn_mask[pad_size:, :pad_size] = True
        # gt cannot see queries
        attn_mask[:pad_size, pad_size:] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), single_pad * 2 * (i + 1) : pad_size] = True
                attn_mask[single_pad * 2 * i : single_pad * 2 * (i + 1), : single_pad * 2 * i] = True

        dn_meta = {
            "pad_size": pad_size,
            "num_dn_group": dn_number,
        }
    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None
        input_label_embed_v2 = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta, input_query_label_v2


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
    post process of dn after output from the transformer
    put the dn part in the dn_meta
    """
    if dn_meta and dn_meta["pad_size"] > 0:
        output_known_class = outputs_class[:, :, : dn_meta["pad_size"], :]
        output_known_coord = outputs_coord[:, :, : dn_meta["pad_size"], :]
        outputs_class = outputs_class[:, :, dn_meta["pad_size"] :, :]
        outputs_coord = outputs_coord[:, :, dn_meta["pad_size"] :, :]
        out = {
            "pred_logits": output_known_class[-1],
            "pred_boxes": output_known_coord[-1],
        }
        if aux_loss:
            out["aux_outputs"] = _set_aux_loss(output_known_class[:-1], output_known_coord[:-1])
        dn_meta["output_known_lbs_bboxes"] = out
    return outputs_class, outputs_coord
