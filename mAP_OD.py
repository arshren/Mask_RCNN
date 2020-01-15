# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from copy import deepcopy
class calc_mAP:
    
    
    def calc_iou( self,gt_bbox, pred_bbox):
        '''
        This function takes the predicted bounding box and ground truth bounding box and 
        return the ioU ratio
        '''
        x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt= gt_bbox
        x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p= pred_bbox

        if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt> y_bottomright_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_topleft_p > x_bottomright_p) or (y_topleft_p> y_bottomright_p):
            raise AssertionError("Predicted Bounding Box is not correct",x_topleft_p, x_bottomright_p,y_topleft_p,y_bottomright_gt)


        #if the GT bbox and predcited BBox do not overlap then iou=0
        if(x_bottomright_gt< x_topleft_p):
            # If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox

            return 0.0
        if(y_bottomright_gt< y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox

            return 0.0
        if(x_topleft_gt> x_bottomright_p): # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox

            return 0.0
        if(y_topleft_gt> y_bottomright_p): # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox

            return 0.0


        GT_bbox_area = (x_bottomright_gt -  x_topleft_gt + 1) * (  y_bottomright_gt -y_topleft_gt + 1)
        Pred_bbox_area =(x_bottomright_p - x_topleft_p + 1 ) * ( y_bottomright_p -y_topleft_p + 1)

        x_top_left =np.max([x_topleft_gt, x_topleft_p])
        y_top_left = np.max([y_topleft_gt, y_topleft_p])
        x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
        y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

        intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left  + 1)

        union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

        return intersection_area/union_area


    def calc_precision_recall(self,img_results):
        """Calculates precision and recall from the set of images
        Args:
            img_results (dict): dictionary formatted like:
                {
                    'img_id1': {'true_pos': int, 'false_pos': int, 'false_neg': int},
                    'img_id2': ...
                    ...
                }
        Returns:
            tuple: of floats of (precision, recall)
        """
        true_pos = 0; false_pos = 0; false_neg = 0
        for _, res in img_results.items():
            true_pos += res['true_pos']
            false_pos += res['false_pos']
            false_neg += res['false_neg']
    
        try:
            precision = true_pos/(true_pos + false_pos)
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = true_pos/(true_pos + false_neg)
        except ZeroDivisionError:
            recall = 0.0
    
        return (precision, recall)



    def get_model_scores(self,pred_boxes):
        """Creates a dictionary of from model_scores to image ids.
        Args:
            pred_boxes (dict): dict of dicts of 'boxes' and 'scores'
        Returns:
            dict: keys are model_scores and values are image ids (usually filenames)
        """
        model_score={}
        for img_id, val in pred_boxes.items():
            for score in val['scores']:
                if score not in model_score.keys():
                    model_score[score]=[img_id]
                else:
                    model_score[score].append(img_id)
        return model_score

    def get_single_image_results(self,gt_b, pred_b, iou_thr):
        """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
        Args:
            gt_boxes (list of list of floats): list of locations of ground truth
                objects as [xmin, ymin, xmax, ymax]
            pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
                and 'scores'
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.
        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """
    
        all_pred_indices = range(len(pred_b))
        all_gt_indices = range(len(gt_b))
        if len(all_pred_indices) == 0:
            tp = 0
            fp = 0
            fn = len(gt_b)
            return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
        if len(all_gt_indices) == 0:
            tp = 0
            fp = len(pred_b)
            fn = 0
            return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}
    
        gt_idx_thr = []
        pred_idx_thr = []
        ious = []
        for ipb, pred_box in enumerate(pred_b):
            for igb, gt_box in enumerate(gt_b):
                iou = self.calc_iou(pred_box, gt_box)
                if iou > iou_thr:
                    gt_idx_thr.append(igb)
                    pred_idx_thr.append(ipb)
                    ious.append(iou)
    
        args_desc = np.argsort(ious)[::-1]
        if len(args_desc) == 0:
            # No matches
            tp = 0
            fp = len(pred_b)
            fn = len(gt_b)
        else:
            gt_match_idx = []
            pred_match_idx = []
            for idx in args_desc:
                gt_idx = gt_idx_thr[idx]
                pr_idx = pred_idx_thr[idx]
                # If the boxes are unmatched, add them to matches
                if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                    gt_match_idx.append(gt_idx)
                    pred_match_idx.append(pr_idx)
            tp = len(gt_match_idx)
            fp = len(pred_b) - len(pred_match_idx)
            fn = len(gt_b) - len(gt_match_idx)
    
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    def get_avg_precision_at_iou(self,gt_boxes, pred_boxes, iou_thr=0.5):
        """Calculates average precision at given IoU threshold.
        Args: Right one
            gt_boxes (list of list of floats): list of locations of ground truth
                objects as [xmin, ymin, xmax, ymax]
            pred_boxes (list of list of floats): list of locations of predicted
                objects as [xmin, ymin, xmax, ymax]
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.
        Returns:
            dict: avg precision as well as summary info about the PR curve
            Keys:
                'avg_prec' (float): average precision for this IoU threshold
                'precisions' (list of floats): precision value for the given
                    model_threshold
                'recall' (list of floats): recall value for given
                    model_threshold
                'models_thrs' (list of floats): model threshold value that
                    precision and recall were computed for.
        """
        model_scores = self.get_model_scores(pred_boxes)
        sorted_model_scores = sorted(model_scores.keys())
    
        # Sort the predicted boxes in descending order (lowest scoring boxes first):
        for img_id in pred_boxes.keys():
            if len(pred_boxes[img_id]) > 0:
                arg_sort = np.argsort(pred_boxes[img_id]['scores'])
                pred_boxes[img_id]['scores'] = np.array(pred_boxes[img_id]['scores'])[arg_sort].tolist()
                pred_boxes[img_id]['boxes'] = np.array(pred_boxes[img_id]['boxes'])[arg_sort].tolist()
    
        pred_boxes_pruned = deepcopy(pred_boxes)
    
        precisions = []
        recalls = []
        model_thrs = []
        img_results = {}
        # Loop over model score thresholds and calculate precision, recall
        for ithr, model_score_thr in enumerate(sorted_model_scores[:-1]):
            # On first iteration, define img_results for the first time:
            img_ids = gt_boxes.keys() if ithr == 0 else model_scores[model_score_thr]
            for img_id in img_ids:
                gt_boxes_img = gt_boxes[img_id]
                if len(pred_boxes_pruned[img_id]) > 0:
                    box_scores = pred_boxes_pruned[img_id]['scores']
                    start_idx = 0
                    for score in box_scores:
                        if score < model_score_thr:
    #                         pred_boxes_pruned[img_id]
                            start_idx += 1
                        else:
                            break
    
                    # Remove boxes, scores of lower than threshold scores:
                    pred_boxes_pruned[img_id]['scores'] = pred_boxes_pruned[img_id]['scores'][start_idx:]
                    pred_boxes_pruned[img_id]['boxes'] = pred_boxes_pruned[img_id]['boxes'][start_idx:]
    
                    # Recalculate image results for this imageprint()
                    print(gt_boxes_img)
                    img_results[img_id] = self.get_single_image_results(
                        gt_boxes_img, pred_boxes_pruned[img_id]['boxes'], iou_thr)
                else:
                    img_results[img_id] = {'true_pos': 0, 'false_pos': 0, 'false_neg': len(gt_boxes_img)}
            prec, rec = self.calc_precision_recall(img_results)
            precisions.append(prec)
            recalls.append(rec)
            model_thrs.append(model_score_thr)
    
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        prec_at_rec = []
        for recall_level in np.linspace(0.0, 1.0, 11):
            try:
                args = np.argwhere(recalls >= recall_level).flatten()
                prec = max(precisions[args])
            except ValueError:
                prec = 0.0
            prec_at_rec.append(prec)
        avg_prec = np.mean(prec_at_rec)
    
        return {
            'avg_prec': avg_prec,
            'precisions': precisions,
            'recalls': recalls,
            'model_thrs': model_thrs}