import cv2
import cvat_sdk.auto_annotation as cvataa
import cvat_sdk.models as sdk_models
import numpy as np
import PIL.Image
import torch
from cellpose import models
from cvat_sdk import make_client

CVAT_URL = "http://localhost:8080"
USERNAME = "admin"
PASSWORD = "admin"
LABEL_NAME = "cell"
TASK_ID = 1


class CellposeDetectionFunction:
    def __init__(self, label_name: str) -> None:
        self.label_name = label_name

        # Load the ML model using the Mac M1 MPS backend
        self._model = models.CellposeModel(
            gpu=True,
            device=torch.device('mps'),
            pretrained_model='cyto3'
        )

    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        # Describe the annotations, mapping your label name to ID 0
        return cvataa.DetectionFunctionSpec(
            labels=[
                cvataa.label_spec(self.label_name, 0, type="polygon")
            ]
        )

    def detect(
        self, context: cvataa.DetectionFunctionContext, image: PIL.Image.Image
    ) -> list[sdk_models.LabeledShapeRequest]:
        img_np = np.array(image)

        # Run the ML model (unpacking 3 values for cellpose>=3.0)
        masks, flows, styles = self._model.eval(
            img_np, diameter=None, channels=[0, 0])

        results = []
        # Iterate through detected instances, ignoring background (0)
        for cell_id in np.unique(masks)[1:]:
            cell_mask = (masks == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) >= 3:
                    # Flatten the array into [x1, y1, x2, y2, ...]
                    points = contour.flatten().tolist()

                    # Convert the results using the SDK's built-in polygon helper
                    results.append(cvataa.polygon(0, points))

        return results


if __name__ == "__main__":
    with make_client(CVAT_URL, credentials=(USERNAME, PASSWORD)) as client:
        func = CellposeDetectionFunction(label_name=LABEL_NAME)
        print(f"Starting annotation for Task {TASK_ID}...")
        cvataa.annotate_task(client, TASK_ID, func)
        print("Annotation complete.")
