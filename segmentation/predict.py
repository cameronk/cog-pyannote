from typing import Any, List 

import numpy as np
from cog import BasePredictor, Input, Path, BaseModel
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
import torch
import logging
import ffmpeg

class AnnotationJson(BaseModel):
    pyannote: str
    uri: str
    modality: str
    content: List[dict]

AVAILABLE_TASKS = ["vad", "osd"]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Check device
        device_count = torch.cuda.device_count()
        if device_count == 0: 
            logging.warn("GPU unavailable!")

        # Load model
        self.model = Model.from_pretrained("pytorch_model.bin")
        logging.info("Completed setup")
        pass
      
    def predict(
        self,
        audio: Path = Input(description="Audio to perform voice activity detection on. Accepts any audio file type convertible to .wav by ffmpeg."),
        task: str = Input(description="Segmentation task to perform", default="vad"),
        # Hyperparameters
        onset: float = Input(description="Onset activation threshold", default=0.8104268538848918),
        offset: float = Input(description="Offset activation threshold", default=0.4806866463041527),
        min_duration_on: float = Input(description="Remove speech regions shorter than that many seconds.", default=0.05537587440407595),
        min_duration_off: float = Input(description="Fill non-speech regions shorter than that many seconds.", default=0.09791355693027545)
    ) -> AnnotationJson:
        if task not in AVAILABLE_TASKS:
            raise Exception("Task %s is not available. Available tasks are: %s" % (task, AVAILABLE_TASKS))

        # Setup hyperparameters
        HYPER_PARAMETERS = {
            "onset": onset,
            "offset": offset,
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off
        }

        # Setup pipeline
        if task == "vad":
            pipeline = VoiceActivityDetection(segmentation=self.model)
        elif task == "osd":
            pipeline = OverlappedSpeechDetection(segmentation=self.model)
        
        pipeline.instantiate(HYPER_PARAMETERS)

        # Convert to wav if necessary
        audio_path = audio
        if audio.suffix != ".wav":
            logging.info("Converting audio to wav")
            audio_path = audio.with_suffix(".wav")
            try: 
                ffmpeg.input(str(audio)).output(str(audio_path)).run()
            except Exception as e:
                logging.exception(e)
                raise e
        
        # Run pipeline
        output = pipeline(audio_path)
        logging.info("Task %s complete" % task)

        data = output.for_json()
        data.update({ "task": task, "parameters": HYPER_PARAMETERS })

        # https://github.com/pyannote/pyannote-core/blob/develop/pyannote/core/annotation.py#L1502
        # {
        #   pyannote: "",
        #   uri: "",
        #   modality: "",
        #   content: [{ segment: { start: 0, end: 1 }, track: "", label: "" }]
        # }
        return data


