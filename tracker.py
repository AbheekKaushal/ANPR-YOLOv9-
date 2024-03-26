from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = 'model_data/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, frame, detections):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-2] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-2] for d in detections]
        class_id = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], class_id[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets)
        self.update_tracks()

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            class_id = track.clid()
            score = track.conf()
            id = track.track_id

            tracks.append(Track(id, bbox, score, class_id))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None
    score = None
    class_id = None

    def __init__(self, id, bbox, score, class_id):
        self.track_id = id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
