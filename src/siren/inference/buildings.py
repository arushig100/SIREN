"""Detect buildings in the pre-image, merge overlapping detections, and score damage
from paired pre- and post-image features."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from siren.perception.model import pool_boxes
from siren.segmentation.building.utils import mask_to_polygons, merge_duplicate_boxes
from siren.settings import BUILDINGS as CONFIG


class BuildingConceptHead(nn.Module):
    """Convolutional classifier for building damage or destruction."""

    def __init__(self):
        """Build the convolutional damage classifier."""
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(384, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Dropout(.3), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x):
        """Compare paired entity features and return two class logits. x is (N, 2, 6272): pre and
        post features, each 128 channels on a 7x7 grid."""
        pre = x[:, 0].reshape(-1, 128, 7, 7)
        post = x[:, 1].reshape(-1, 128, 7, 7)
        # Convolve pre, post and their difference (384 channels), then pool mean and max.
        h = self.conv(torch.cat((pre, post, post - pre), 1))
        return self.classifier(torch.cat((h.mean((2, 3)), h.amax((2, 3))), 1))


def load_head(path, device):
    """Load a damage or destruction head saved as {"model_state_dict": ...}."""
    head = BuildingConceptHead().to(device)
    head.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model_state_dict'])
    return head.eval()


def pool_building_features(fpn, boxes):
    """Pool 7x7 features of building boxes with the fixed context margin: (N, 6272)."""
    return pool_boxes(fpn, boxes, (CONFIG.context_pad, 0.0))


def merge_detections(candidates):
    """Remove duplicate detections from the full view and the tiles; keep the higher score."""
    kept = merge_duplicate_boxes([c['box'] for c in candidates], [c['score'] for c in candidates],
                                 iou=CONFIG.merge_iou, containment=CONFIG.merge_containment)
    return [candidates[i] for i in kept]


def detect_buildings(model, image, x0, y0, size, view, score_min):
    """Detect buildings in a 1024-pixel view of the image region at (x0, y0, size).

    Returns the view's FPN features and one dict per kept detection. Boxes and areas are in
    native image pixels. Each crop is the mask inside its box, in view pixels; org (crop corner
    in the view), k (scale) and off (region corner) map it back."""
    S = CONFIG.image_size
    det = model.det
    # The detector's stages are run one by one so the FPN features can be returned.
    images, _ = det.transform([image], None)
    fpn = model.fpn(images.tensors)
    proposals, _ = det.rpn(images, fpn, None)
    detections, _ = det.roi_heads(fpn, proposals, images.image_sizes, None)
    detection = det.transform.postprocess(detections, images.image_sizes, [(S, S)])[0]
    # Label 1 is the building class; threshold its predicted mask.
    keep = (detection['labels'] == 1) & (detection['scores'] >= score_min)
    scores = detection['scores'][keep]
    masks = detection['masks'][keep][:, 0] >= CONFIG.mask_threshold
    output = []
    if masks.shape[0]:
        area = masks.sum((1, 2)).float()
        # Mask extent: argmax returns the first occupied column (or row), and on the
        # flipped mask the last one, so x1 and y1 are exclusive (last pixel + 1).
        cols, rows = masks.any(1), masks.any(2)
        bx0 = cols.float().argmax(1).float()
        bx1 = S - cols.flip(1).float().argmax(1).float()
        by0 = rows.float().argmax(1).float()
        by1 = S - rows.flip(1).float().argmax(1).float()
        scale = size / S
        # Drop masks within CONFIG.tile_edge_margin view pixels of a tile edge inside the image: such
        # buildings are only partly visible, and the full view or an overlapping tile can
        # detect them whole. Edges on the image border keep their detections.
        margin = CONFIG.tile_edge_margin
        truncated = (((bx0 <= margin) & (x0 > 0)) | ((by0 <= margin) & (y0 > 0))
                     | ((bx1 >= S - margin) & (x0 + size < S))
                     | ((by1 >= S - margin) & (y0 + size < S)))
        # scale**2 converts view pixels to native image pixels (1/4 for a 512-px tile).
        valid = (area * scale * scale >= CONFIG.min_mask_area) & (area > 0) & ~truncated
        for j in torch.nonzero(valid)[:, 0].tolist():
            X0, Y0, X1, Y1 = int(bx0[j]), int(by0[j]), int(bx1[j]), int(by1[j])
            output.append(dict(
                view=view, score=float(scores[j]),
                box=[X0 * scale + x0, Y0 * scale + y0, X1 * scale + x0, Y1 * scale + y0],
                area=float(area[j] * scale * scale),
                crop=masks[j, Y0:Y1, X0:X1].cpu().numpy().astype(np.uint8),
                org=(X0, Y0), k=scale, off=(x0, y0)))
    return fpn, output


def upsampled_tile(image, x0, y0, size):
    """Crop a tile of the image and resize it to the 1024-pixel detector input."""
    return F.interpolate(image[None, :, y0:y0 + size, x0:x0 + size], size=(CONFIG.image_size,) * 2,
                         mode='bilinear', align_corners=False)[0].clamp(0, 1)


class BuildingPipeline:
    """Detect buildings and score damage and destruction with a loaded model and two heads.

    Inputs are paired RGB tensors, 3x1024x1024 in [0, 1]. The class reads and writes no files.
    """

    def __init__(self, model, heads):
        """Attach the shared encoder, building detector and trained damage classifiers."""
        self.model = model.eval()
        self.det = model.det
        self.det.roi_heads.score_thresh = CONFIG.detector_score_floor
        self.thresholds = {'is_damaged': CONFIG.damaged_threshold,
                           'is_destroyed': CONFIG.destroyed_threshold}
        if set(heads) != set(self.thresholds):
            raise ValueError('Both the damage and destruction heads are required')
        self.heads = {key: head.eval() for key, head in heads.items()}

    def detect(self, image, x0, y0, size, view, score_min):
        """Detect buildings in one view; see detect_buildings."""
        return detect_buildings(self.model, image, x0, y0, size, view, score_min)

    def pooled(self, fpn, boxes):
        """Pool building features with the fixed context margin: (N, 6272)."""
        return pool_building_features(fpn, boxes)

    @torch.inference_mode()
    def run(self, pre, post, *, chip_id, split, disaster_id=''):
        """Return the building record and the full-image pre and post FPN features.

        Post features are None when no building is found; tile features are never returned."""
        if pre.shape != (3, CONFIG.image_size, CONFIG.image_size) or post.shape != pre.shape:
            raise ValueError('Building inference requires paired 3x1024x1024 tensors')
        for image in (pre, post):
            if image.dtype != torch.float32:
                raise ValueError('Building inference requires float32 tensors')
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError('Image values must be finite and in [0, 1]')
        fpn_pre, full = self.detect(pre, 0, 0, CONFIG.image_size, 0, CONFIG.full_score)
        base = [c for c in full if c['score'] >= CONFIG.base_score]
        tiled = 0
        # Dense scenes combine the full view with tiles containing enough building centers.
        # Other scenes use only the confident full-view detections.
        if len(base) >= CONFIG.tile_trigger:
            candidates = list(full)
            cx = [(c['box'][0] + c['box'][2]) / 2 for c in base]
            cy = [(c['box'][1] + c['box'][3]) / 2 for c in base]
            size = CONFIG.tile_size
            # View 0 is the full image. Views 1 to 9 are the nine 512-pixel tiles in row order,
            # each upsampled to 1024 pixels for detection.
            for index, (x0, y0) in enumerate(CONFIG.tile_origins()):
                inside = sum(1 for x, y in zip(cx, cy) if x0 <= x < x0 + size and y0 <= y < y0 + size)
                if inside < CONFIG.tile_min_buildings:
                    continue
                tiled += 1
                tile = upsampled_tile(pre, x0, y0, size)
                candidates += self.detect(tile, x0, y0, size, CONFIG.tile_view_start + index,
                                          CONFIG.tile_score)[1]
        else:
            candidates = base
        candidates = merge_detections(candidates)
        entities = []
        fpn_post = None
        if candidates:
            boxes = torch.tensor([c['box'] for c in candidates], device=pre.device, dtype=torch.float32)
            # Both images are pooled at the same boxes from full-image features: (N, 2, 6272).
            pooled_pre = self.pooled(fpn_pre, boxes)
            fpn_post = self.model.fpn(post[None])
            features = torch.stack((pooled_pre, self.pooled(fpn_post, boxes)), 1)
            # Probability of the positive class (damaged or destroyed) for every building.
            scores = {q: self.heads[q](features).softmax(-1)[:, 1].cpu().tolist() for q in self.thresholds}
            for i, candidate in enumerate(candidates):
                # Trace polygons on the view-pixel crop, then map them to native image pixels.
                polygons = mask_to_polygons(candidate['crop'], simplify_eps=2., min_area=0.)
                scale = candidate['k']
                polygons = [[[(x + candidate['org'][0]) * scale + candidate['off'][0],
                              (y + candidate['org'][1]) * scale + candidate['off'][1]]
                             for x, y in ring] for ring in polygons]
                if not polygons:  # fall back to the box outline
                    b = candidate['box']
                    polygons = [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]]
                attributes = {'detection_score': candidate['score']}
                for q, threshold in self.thresholds.items():
                    attributes[q + '_score'] = scores[q][i]
                    attributes[q] = scores[q][i] >= threshold
                entities.append({
                    'id': f'building:{i + 1:06d}', 'type': 'building',
                    'source': {'method': 'maskrcnn_satlas_adaptive_tiles',
                               'view': candidate['view'], 'score': candidate['score']},
                    'geometry': {'bbox_xyxy': candidate['box'], 'polygon_xy': polygons,
                                 'area_px': candidate['area']}, 'attributes': attributes})
        # The export block records the detection settings and the number of tiles processed.
        record = {'schema': 'siren_scene', 'chip_id': chip_id, 'disaster_id': disaster_id,
                  'split': split, 'entities': {'buildings': entities},
                  'export': {'pipeline': 'siren',
                             'tile_min_buildings': CONFIG.tile_min_buildings, 'tiled': tiled,
                             'thresholds': self.thresholds, 'full_score': CONFIG.full_score,
                             'tile_score': CONFIG.tile_score, 'tile_trigger': CONFIG.tile_trigger}}
        return record, fpn_pre, fpn_post
