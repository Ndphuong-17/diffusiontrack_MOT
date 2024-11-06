# diffusiontrack_MOT
 


**Mode Descriptions**:

- **Eval mode = False**: Used for training models that predict object movement.
    - **Original Model**: `D2MP_model`
    - **New Model**: `UNET3Plus` combined with `StableDiffusion` (by Huy)

- **Eval mode = True**: Used for evaluating performance by combining the trained model with a detector and tracking module.
    - **Online**: For live or real-time scenarios where frames are processed sequentially as they come in.
    - **Offline**: For recorded video where all frames are available beforehand.

**Detector**
   - **Supported Models**:
       - `YOLOX`, `YOLO10`: (x)
       - `YOLO8`          : (v)
   
   - **Function**:
       - **Input**: Every nth frame from the video.
       - **Output**: Bounding box locations of objects detected in the nth frame along with feature representations of each detected object.

**Predict Movement**
   - **Status**: Currently in design and experimentation phase (not fully implemented).
   - **Objective**: To predict the movement of detected objects between frames.

   - **Details**:
       - **Label**: The difference in object location between the nth and (n-1)th frames, with dimensions `(batch size * 4)`.
       - **Input**: Bounding box locations from the (n-interval)th frame to the (n-1)th frame. Each set of boxes has dimensions `(batch size * interval * 8)`.
       - **Output**: Predicted movement (delta location) between the nth frame and (n-1)th frame with dimensions `(batch size * 4)`.

**Re-Identification (Re-Id)**
   - **Objective**: Match objects across frames based on their predicted locations and features.

   - **Details**:
       - **Input**:
           - Bounding box locations for objects detected in the nth frame `(n_objects, 4)`.
           - Feature vectors for each object in the nth frame, provided by the detector `(n_objects, features)`. For YOLO8, feature size is 64.
           - Predicted locations of objects in the nth frame `(n_objects, 4)`.

        - **Output**: Matched object IDs for tracking across frames.

   - **Steps**:
       1. **Handle High-Confidence Detections**:
       2. **Handle Low-Confidence Detections**:
       3. **Match Predicted and Detected Locations to update ID**:
        - Existence object
        - Existence but blurred object
        - NotExistence object
       4. **Update Tracking Information**: