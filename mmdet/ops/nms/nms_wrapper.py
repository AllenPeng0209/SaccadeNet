import numpy as np
import torch

from . import nms_cuda, nms_cpu
from .soft_nms_cpu import soft_nms_cpu
from IPython import embed

def nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th, iou_thr)
        else:
            inds = nms_cpu.nms(dets_th, iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_np = dets.detach().cpu().numpy()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_np = dets
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    new_dets, inds = soft_nms_cpu(
        dets_np,
        iou_thr,
        method=method_codes[method],
        sigma=sigma,
        min_score=min_score)

    if is_tensor:
        return dets.new_tensor(new_dets), dets.new_tensor(
            inds, dtype=torch.long)
    else:
        return new_dets.astype(np.float32), inds.astype(np.int64)



def saccade_nms(dets, iou_thr, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
    # TODO write saccade_nms in cuda and cpu entension.
        
        left = dets[:,0]
        top  = dets[:,1]
        right = dets[:,2]
        bottom = dets[:,3]
        cls_score = dets[:,4]
        box_iou_score = dets[:,5]
        left_score = dets[:,6]
        top_score  = dets[:,7]
        right_score = dets[:,8]
        bottom_score = dets[:,9]
        area = (right-left)*(bottom-top)
        order = (cls_score*box_iou_score).argsort(descending=True)
        bbox_num=0
        unmerge_list={}
        keep=[]
        bbox_list=[]
        while order.size()[0]>0:
            i=order[0]
            left_ = torch.max(left[i],left[order[1:]])
            top_ = torch.max(top[i],top[order[1:]])
            right_ = torch.min(right[i],right[order[1:]])
            bottom_ = torch.min(bottom[i],bottom[order[1:]])       
            inter= torch.max(torch.zeros(1, device="cuda:0"),right_-left_)*torch.max(torch.zeros(1, device="cuda:0"),bottom_-top_)
            iou = inter / (area[i]+area[order[:1]]-inter)
            #merge_ids included self and iou > thr
            unmerge_ids = order[1:][(iou>iou_thr)].tolist()
            unmerge_ids.append(i.item())
            unmerge_list[bbox_num] = unmerge_ids
            order =order[1:][(iou<=iou_thr)]
            bbox_num+=1
    for i in range(bbox_num):
        left_ids =  (dets[unmerge_list[i]][:,[4,5]].sum(dim=1)/2).max(0)
        top_ids =   (dets[unmerge_list[i]][:,[4,5]].sum(dim=1)/2).max(0)
        right_ids =  (dets[unmerge_list[i]][:,[4,5]].sum(dim=1)/2).max(0)
        bottom_ids =  (dets[unmerge_list[i]][:,[4,5]].sum(dim=1)/2).max(0)
        left = dets[unmerge_list[i]][left_ids[1]][0]
        top =  dets[unmerge_list[i]][top_ids[1]][1]
        right =  dets[unmerge_list[i]][right_ids[1]][2]
        bottom =  dets[unmerge_list[i]][bottom_ids[1]][3]
        score = ( left_ids[0] + top_ids[0] + right_ids[0] + bottom_ids[0]
                ) / 4
        bbox = torch.stack([left,top,right,bottom,score])
        bbox_list.append(bbox)
    bbox_list= torch.stack(bbox_list)
    # _ just to fill ids, our ids already be merged.
    _ = 0
    return bbox_list , _




    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

